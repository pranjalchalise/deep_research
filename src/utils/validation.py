# src/utils/validation.py
"""
Input validation utilities for robust state access and data handling.

Provides safe accessors and validators to prevent crashes from:
- Missing or None values
- Wrong types
- Empty collections
- Malformed data structures
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Safe State Access
# =============================================================================

def get_str(state: Dict, key: str, default: str = "") -> str:
    """Safely get a string from state."""
    value = state.get(key)
    if value is None:
        return default
    if isinstance(value, str):
        return value
    # Try to convert
    try:
        return str(value)
    except Exception:
        logger.warning(f"Could not convert {key}={value!r} to string, using default")
        return default


def get_int(state: Dict, key: str, default: int = 0, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
    """Safely get an integer from state with optional bounds."""
    value = state.get(key)
    if value is None:
        return default
    try:
        result = int(value)
        if min_val is not None and result < min_val:
            logger.warning(f"{key}={result} below min {min_val}, clamping")
            result = min_val
        if max_val is not None and result > max_val:
            logger.warning(f"{key}={result} above max {max_val}, clamping")
            result = max_val
        return result
    except (ValueError, TypeError):
        logger.warning(f"Could not convert {key}={value!r} to int, using default {default}")
        return default


def get_float(state: Dict, key: str, default: float = 0.0, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
    """Safely get a float from state with optional bounds."""
    value = state.get(key)
    if value is None:
        return default
    try:
        result = float(value)
        if min_val is not None and result < min_val:
            result = min_val
        if max_val is not None and result > max_val:
            result = max_val
        return result
    except (ValueError, TypeError):
        logger.warning(f"Could not convert {key}={value!r} to float, using default {default}")
        return default


def get_bool(state: Dict, key: str, default: bool = False) -> bool:
    """Safely get a boolean from state."""
    value = state.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "yes", "1", "on")
    return bool(value)


def get_list(state: Dict, key: str, default: Optional[List] = None) -> List:
    """Safely get a list from state."""
    if default is None:
        default = []
    value = state.get(key)
    if value is None:
        return default
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    logger.warning(f"{key} is not a list (got {type(value).__name__}), using default")
    return default


def get_dict(state: Dict, key: str, default: Optional[Dict] = None) -> Dict:
    """Safely get a dict from state."""
    if default is None:
        default = {}
    value = state.get(key)
    if value is None:
        return default
    if isinstance(value, dict):
        return value
    logger.warning(f"{key} is not a dict (got {type(value).__name__}), using default")
    return default


def get_non_empty_list(state: Dict, key: str) -> Optional[List]:
    """Get a list only if it's non-empty, else return None."""
    value = get_list(state, key, default=[])
    return value if value else None


def get_first_message_content(messages: List, default: str = "") -> str:
    """Safely get content from the first message in a list."""
    if not messages:
        return default
    msg = messages[0]
    if hasattr(msg, "content"):
        return str(msg.content) if msg.content else default
    return default


def get_last_message_content(messages: List, default: str = "") -> str:
    """Safely get content from the last message in a list."""
    if not messages:
        return default
    msg = messages[-1]
    if hasattr(msg, "content"):
        return str(msg.content) if msg.content else default
    return default


def get_human_message_content(messages: List, default: str = "") -> str:
    """Safely get content from the last human message in a list."""
    if not messages:
        return default

    for msg in reversed(messages):
        # Check for LangChain message types
        if hasattr(msg, "type") and msg.type == "human":
            return str(msg.content) if msg.content else default
        # Check class name
        if "HumanMessage" in type(msg).__name__:
            return str(msg.content) if msg.content else default

    # Fallback to last message
    return get_last_message_content(messages, default)


# =============================================================================
# Type Validation
# =============================================================================

def ensure_list(value: Any, item_type: Optional[Type] = None) -> List:
    """
    Ensure value is a list, converting if possible.

    Args:
        value: Value to convert
        item_type: If provided, filter items to only this type

    Returns:
        List (possibly empty)
    """
    if value is None:
        return []

    if isinstance(value, list):
        result = value
    elif isinstance(value, (tuple, set)):
        result = list(value)
    elif isinstance(value, dict):
        result = [value]  # Wrap single dict in list
    else:
        result = [value]  # Wrap scalar in list

    if item_type is not None:
        result = [item for item in result if isinstance(item, item_type)]

    return result


def ensure_dict(value: Any) -> Dict:
    """Ensure value is a dict, returning empty dict if not."""
    if isinstance(value, dict):
        return value
    return {}


def ensure_str(value: Any, default: str = "") -> str:
    """Ensure value is a string, converting if possible."""
    if value is None:
        return default
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:
        return default


# =============================================================================
# Dict Key Validation
# =============================================================================

def safe_get(d: Dict, *keys: str, default: Any = None) -> Any:
    """
    Safely get a nested value from a dict.

    Example:
        safe_get(data, "foo", "bar", "baz", default=0)
        # Equivalent to data.get("foo", {}).get("bar", {}).get("baz", 0)
    """
    current = d
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def has_keys(d: Dict, *required_keys: str) -> bool:
    """Check if dict has all required keys with non-None values."""
    if not isinstance(d, dict):
        return False
    return all(d.get(key) is not None for key in required_keys)


def validate_dict_keys(
    d: Dict,
    required: Optional[List[str]] = None,
    optional: Optional[List[str]] = None,
    name: str = "dict",
) -> List[str]:
    """
    Validate dict has required keys, return list of missing keys.

    Args:
        d: Dict to validate
        required: Keys that must be present
        optional: Keys that may be present (for documentation)
        name: Name for logging

    Returns:
        List of missing required keys (empty if valid)
    """
    if not isinstance(d, dict):
        logger.warning(f"{name} is not a dict")
        return required or []

    missing = []
    for key in (required or []):
        if key not in d or d[key] is None:
            missing.append(key)

    if missing:
        logger.warning(f"{name} missing required keys: {missing}")

    return missing


# =============================================================================
# List Item Validation
# =============================================================================

def validate_list_items(
    items: List,
    required_keys: List[str],
    name: str = "item",
) -> List[Dict]:
    """
    Filter list to only items with required keys.

    Returns items that have all required keys with non-None values.
    """
    valid = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            logger.debug(f"{name}[{i}] is not a dict, skipping")
            continue

        missing = [k for k in required_keys if k not in item or item[k] is None]
        if missing:
            logger.debug(f"{name}[{i}] missing keys {missing}, skipping")
            continue

        valid.append(item)

    if len(valid) < len(items):
        logger.debug(f"Filtered {len(items) - len(valid)} invalid {name}s")

    return valid


def get_valid_items(
    state: Dict,
    key: str,
    required_keys: List[str],
) -> List[Dict]:
    """
    Get list from state and filter to valid items.

    Convenience function combining get_list and validate_list_items.
    """
    items = get_list(state, key, default=[])
    return validate_list_items(items, required_keys, name=key)


# =============================================================================
# Safe Operations
# =============================================================================

def safe_slice(lst: List, start: int = 0, end: Optional[int] = None) -> List:
    """Safely slice a list without index errors."""
    if not lst:
        return []
    if end is None:
        return lst[start:]
    return lst[start:end]


def safe_first(lst: List, default: Any = None) -> Any:
    """Safely get first item of a list."""
    return lst[0] if lst else default


def safe_last(lst: List, default: Any = None) -> Any:
    """Safely get last item of a list."""
    return lst[-1] if lst else default


def safe_index(lst: List, idx: int, default: Any = None) -> Any:
    """Safely get item at index."""
    try:
        return lst[idx] if lst else default
    except IndexError:
        return default


def safe_join(items: List, separator: str = ", ", max_items: Optional[int] = None) -> str:
    """Safely join list items as strings."""
    if not items:
        return ""

    str_items = [str(item) for item in items if item is not None]

    if max_items and len(str_items) > max_items:
        str_items = str_items[:max_items]
        str_items.append(f"... and {len(items) - max_items} more")

    return separator.join(str_items)


# =============================================================================
# Assertion Helpers (for clear error messages)
# =============================================================================

def require_non_empty(value: Any, name: str = "value") -> None:
    """Raise ValueError if value is empty/None."""
    if not value:
        raise ValueError(f"{name} cannot be empty")


def require_type(value: Any, expected_type: Type, name: str = "value") -> None:
    """Raise TypeError if value is not expected type."""
    if not isinstance(value, expected_type):
        raise TypeError(f"{name} must be {expected_type.__name__}, got {type(value).__name__}")


def require_keys(d: Dict, *keys: str, name: str = "dict") -> None:
    """Raise ValueError if dict is missing required keys."""
    missing = [k for k in keys if k not in d or d[k] is None]
    if missing:
        raise ValueError(f"{name} missing required keys: {missing}")


__all__ = [
    # State access
    "get_str",
    "get_int",
    "get_float",
    "get_bool",
    "get_list",
    "get_dict",
    "get_non_empty_list",
    "get_first_message_content",
    "get_last_message_content",
    "get_human_message_content",
    # Type validation
    "ensure_list",
    "ensure_dict",
    "ensure_str",
    # Dict validation
    "safe_get",
    "has_keys",
    "validate_dict_keys",
    # List validation
    "validate_list_items",
    "get_valid_items",
    # Safe operations
    "safe_slice",
    "safe_first",
    "safe_last",
    "safe_index",
    "safe_join",
    # Assertions
    "require_non_empty",
    "require_type",
    "require_keys",
]
