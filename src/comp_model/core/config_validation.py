"""Shared helpers for strict declarative config validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def validate_allowed_keys(
    mapping: Mapping[str, Any],
    *,
    field_name: str,
    allowed_keys: Iterable[str],
) -> None:
    """Validate that a mapping only contains allowed keys.

    Parameters
    ----------
    mapping : Mapping[str, Any]
        Configuration mapping to validate.
    field_name : str
        Human-readable path used in error messages.
    allowed_keys : Iterable[str]
        Allowed key names for ``mapping``.

    Raises
    ------
    ValueError
        If unknown keys are present.
    """

    allowed = set(str(key) for key in allowed_keys)
    unknown = sorted(str(key) for key in mapping if str(key) not in allowed)
    if unknown:
        raise ValueError(f"{field_name} has unknown keys: {unknown}")


def validate_required_keys(
    mapping: Mapping[str, Any],
    *,
    field_name: str,
    required_keys: Iterable[str],
) -> None:
    """Validate that required keys are present in a mapping.

    Parameters
    ----------
    mapping : Mapping[str, Any]
        Configuration mapping to validate.
    field_name : str
        Human-readable path used in error messages.
    required_keys : Iterable[str]
        Keys that must be present in ``mapping``.

    Raises
    ------
    ValueError
        If required keys are missing.
    """

    required = set(str(key) for key in required_keys)
    missing = sorted(key for key in required if key not in mapping)
    if missing:
        raise ValueError(f"{field_name} is missing required keys: {missing}")


__all__ = ["validate_allowed_keys", "validate_required_keys"]

