"""Shared config parsers for model component specifications."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from comp_model.core.config_validation import validate_allowed_keys


@dataclass(frozen=True, slots=True)
class ModelComponentSpec:
    """Model component spec parsed from config.

    Parameters
    ----------
    component_id : str
        Model component ID in plugin registry.
    kwargs : dict[str, Any]
        Fixed model constructor kwargs.
    """

    component_id: str
    kwargs: dict[str, Any]


def model_component_spec_from_config(model_cfg: Mapping[str, Any]) -> ModelComponentSpec:
    """Parse model component spec from config mapping."""

    mapping = _require_mapping(model_cfg, field_name="model")
    validate_allowed_keys(mapping, field_name="model", allowed_keys=("component_id", "kwargs"))
    component_id = _coerce_non_empty_str(mapping.get("component_id"), field_name="model.component_id")
    kwargs = _require_mapping(mapping.get("kwargs", {}), field_name="model.kwargs")
    return ModelComponentSpec(component_id=component_id, kwargs=dict(kwargs))


def _require_mapping(raw: Any, *, field_name: str) -> Mapping[str, Any]:
    """Require a JSON-like mapping object."""

    if not isinstance(raw, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return raw


def _coerce_non_empty_str(raw: Any, *, field_name: str) -> str:
    """Coerce a required non-empty string."""

    if raw is None:
        raise ValueError(f"{field_name} must be a non-empty string")
    value = str(raw).strip()
    if not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


__all__ = ["ModelComponentSpec", "model_component_spec_from_config"]
