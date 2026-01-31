# src/utils/json_utils.py
"""
Robust JSON parsing utilities for LLM outputs.

LLMs often return JSON wrapped in markdown, with trailing commas,
or with other formatting issues. This module provides helpers to
extract and parse JSON reliably.
"""
from __future__ import annotations

import json
import re
from typing import Any, List, Optional, Type, TypeVar, Union

T = TypeVar("T")


class JSONParseError(Exception):
    """Raised when JSON parsing fails after all attempts."""

    def __init__(self, message: str, raw_text: str, attempts: List[str]):
        super().__init__(message)
        self.raw_text = raw_text
        self.attempts = attempts


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ``` or ``` ... ```)."""
    text = text.strip()

    # Pattern for ```json ... ``` or ```JSON ... ``` or just ``` ... ```
    fence_pattern = r"^```(?:json|JSON)?\s*\n?(.*?)\n?```$"
    match = re.match(fence_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Also handle case where fences are not at start/end
    # Find content between first ``` and last ```
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            # Take the content between first and last fence
            # parts[0] = before first ```, parts[1] = inside, parts[2] = after
            inner = parts[1]
            # Remove optional language identifier on first line
            lines = inner.split("\n", 1)
            if len(lines) > 1 and lines[0].strip().lower() in ("json", ""):
                return lines[1].strip()
            return inner.strip()

    return text


def _extract_json_structure(text: str, start_char: str, end_char: str) -> Optional[str]:
    """
    Extract a JSON structure (object or array) by finding balanced brackets.
    Returns None if no valid structure found.
    """
    start_idx = text.find(start_char)
    if start_idx == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start_idx, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == start_char:
            depth += 1
        elif char == end_char:
            depth -= 1
            if depth == 0:
                return text[start_idx : i + 1]

    return None


def _fix_common_issues(text: str) -> str:
    """Fix common JSON formatting issues from LLMs."""
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # Fix single quotes used instead of double quotes (careful with apostrophes)
    # Only do this if there are no double quotes at all
    if '"' not in text and "'" in text:
        # Simple replacement - may not work for all cases
        text = text.replace("'", '"')

    # Remove JavaScript-style comments
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    # Fix unquoted keys (simple cases only)
    # e.g., {key: "value"} -> {"key": "value"}
    text = re.sub(r"{\s*(\w+)\s*:", r'{"\1":', text)
    text = re.sub(r",\s*(\w+)\s*:", r',"\1":', text)

    return text


def parse_json(
    text: str,
    expected_type: Optional[Type[T]] = None,
    default: Optional[T] = None,
) -> Union[T, Any]:
    """
    Parse JSON from LLM output with multiple fallback strategies.

    Args:
        text: Raw text from LLM that should contain JSON
        expected_type: Expected type (list or dict) to guide extraction
        default: Value to return if all parsing fails (if None, raises JSONParseError)

    Returns:
        Parsed JSON as dict or list

    Raises:
        JSONParseError: If parsing fails and no default provided
    """
    if not text or not text.strip():
        if default is not None:
            return default
        raise JSONParseError("Empty input", text or "", [])

    attempts: List[str] = []
    original = text

    # Step 1: Strip markdown fences
    text = _strip_markdown_fences(text)
    attempts.append("strip_markdown")

    # Step 2: Try direct parse
    try:
        result = json.loads(text)
        if expected_type is None or isinstance(result, expected_type):
            return result
    except json.JSONDecodeError:
        pass
    attempts.append("direct_parse")

    # Step 3: Try to extract JSON object or array
    if expected_type is list or (expected_type is None and "[" in text):
        extracted = _extract_json_structure(text, "[", "]")
        if extracted:
            try:
                result = json.loads(extracted)
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass
        attempts.append("extract_array")

    if expected_type is dict or (expected_type is None and "{" in text):
        extracted = _extract_json_structure(text, "{", "}")
        if extracted:
            try:
                result = json.loads(extracted)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass
        attempts.append("extract_object")

    # Step 4: Fix common issues and retry
    fixed = _fix_common_issues(text)
    if fixed != text:
        try:
            result = json.loads(fixed)
            if expected_type is None or isinstance(result, expected_type):
                return result
        except json.JSONDecodeError:
            pass

        # Try extraction again on fixed text
        if expected_type is list or expected_type is None:
            extracted = _extract_json_structure(fixed, "[", "]")
            if extracted:
                try:
                    return json.loads(extracted)
                except json.JSONDecodeError:
                    pass

        if expected_type is dict or expected_type is None:
            extracted = _extract_json_structure(fixed, "{", "}")
            if extracted:
                try:
                    return json.loads(extracted)
                except json.JSONDecodeError:
                    pass
    attempts.append("fix_common_issues")

    # Step 5: Last resort - try to find any JSON-like structure
    # Look for array pattern
    array_match = re.search(r"\[[\s\S]*\]", text)
    if array_match:
        try:
            return json.loads(array_match.group())
        except json.JSONDecodeError:
            pass

    # Look for object pattern
    obj_match = re.search(r"\{[\s\S]*\}", text)
    if obj_match:
        try:
            return json.loads(obj_match.group())
        except json.JSONDecodeError:
            pass
    attempts.append("regex_fallback")

    # All attempts failed
    if default is not None:
        return default

    raise JSONParseError(
        f"Failed to parse JSON after {len(attempts)} attempts",
        original,
        attempts,
    )


def parse_json_array(text: str, default: Optional[List] = None) -> List:
    """Parse JSON expecting a list/array."""
    result = parse_json(text, expected_type=list, default=default)
    if isinstance(result, list):
        return result
    if default is not None:
        return default
    raise JSONParseError("Expected array but got " + type(result).__name__, text, [])


def parse_json_object(text: str, default: Optional[dict] = None) -> dict:
    """Parse JSON expecting a dict/object."""
    result = parse_json(text, expected_type=dict, default=default)
    if isinstance(result, dict):
        return result
    if default is not None:
        return default
    raise JSONParseError("Expected object but got " + type(result).__name__, text, [])


__all__ = [
    "JSONParseError",
    "parse_json",
    "parse_json_array",
    "parse_json_object",
]
