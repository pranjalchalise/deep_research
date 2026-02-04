"""
Forgiving JSON parser for LLM outputs.

LLMs love wrapping JSON in markdown fences, leaving trailing commas,
using single quotes, or mixing prose with JSON. We try progressively
harder strategies to pull valid JSON out of that mess.
"""
from __future__ import annotations

import json
import re
from typing import Any, List, Optional, Type, TypeVar, Union

T = TypeVar("T")


class JSONParseError(Exception):
    """Raised when we can't extract JSON after exhausting all strategies."""

    def __init__(self, message: str, raw_text: str, attempts: List[str]):
        super().__init__(message)
        self.raw_text = raw_text
        self.attempts = attempts


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers that LLMs almost always add."""
    text = text.strip()

    fence_pattern = r"^```(?:json|JSON)?\s*\n?(.*?)\n?```$"
    match = re.match(fence_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Sometimes the fences are buried in surrounding prose
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            inner = parts[1]
            lines = inner.split("\n", 1)
            if len(lines) > 1 and lines[0].strip().lower() in ("json", ""):
                return lines[1].strip()
            return inner.strip()

    return text


def _extract_json_structure(text: str, start_char: str, end_char: str) -> Optional[str]:
    """Walk the string to find a balanced bracket pair, respecting nesting and quoted strings."""
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
    """Patch up the most common LLM JSON mistakes so json.loads has a chance."""
    # Trailing commas before closing brackets
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # Single quotes -> double quotes, but only when there are no double quotes
    # at all (otherwise we'd mangle apostrophes in normal strings)
    if '"' not in text and "'" in text:
        text = text.replace("'", '"')

    # Strip JS-style comments
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    # Unquoted keys: {key: "val"} -> {"key": "val"}
    text = re.sub(r"{\s*(\w+)\s*:", r'{"\1":', text)
    text = re.sub(r",\s*(\w+)\s*:", r',"\1":', text)

    return text


def parse_json(
    text: str,
    expected_type: Optional[Type[T]] = None,
    default: Optional[T] = None,
) -> Union[T, Any]:
    """
    Try to parse JSON from messy LLM output using multiple strategies.

    Pass expected_type=list or dict to guide extraction when there are
    multiple JSON blobs in the text. Returns default on failure, or
    raises JSONParseError if no default was given.
    """
    if not text or not text.strip():
        if default is not None:
            return default
        raise JSONParseError("Empty input", text or "", [])

    attempts: List[str] = []
    original = text

    # Step 1: strip markdown fences
    text = _strip_markdown_fences(text)
    attempts.append("strip_markdown")

    # Step 2: try direct parse
    try:
        result = json.loads(text)
        if expected_type is None or isinstance(result, expected_type):
            return result
    except json.JSONDecodeError:
        pass
    attempts.append("direct_parse")

    # Step 3: bracket-matching extraction
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

    # Step 4: fix common quirks and retry
    fixed = _fix_common_issues(text)
    if fixed != text:
        try:
            result = json.loads(fixed)
            if expected_type is None or isinstance(result, expected_type):
                return result
        except json.JSONDecodeError:
            pass

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

    # Step 5: last resort -- greedy regex grab
    array_match = re.search(r"\[[\s\S]*\]", text)
    if array_match:
        try:
            return json.loads(array_match.group())
        except json.JSONDecodeError:
            pass

    obj_match = re.search(r"\{[\s\S]*\}", text)
    if obj_match:
        try:
            return json.loads(obj_match.group())
        except json.JSONDecodeError:
            pass
    attempts.append("regex_fallback")

    if default is not None:
        return default

    raise JSONParseError(
        f"Failed to parse JSON after {len(attempts)} attempts",
        original,
        attempts,
    )


def parse_json_array(text: str, default: Optional[List] = None) -> List:
    """Shorthand -- parse and expect a list back."""
    result = parse_json(text, expected_type=list, default=default)
    if isinstance(result, list):
        return result
    if default is not None:
        return default
    raise JSONParseError("Expected array but got " + type(result).__name__, text, [])


def parse_json_object(text: str, default: Optional[dict] = None) -> dict:
    """Shorthand -- parse and expect a dict back."""
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
