"""Requirement descriptors for model/problem compatibility.

The requirement object is intentionally lightweight. It can be attached to
component manifests and consumed by inference-time compatibility checks.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ComponentRequirements:
    """Data-field requirements for replay and inference.

    Parameters
    ----------
    required_observation_fields : tuple[str, ...], optional
        Observation keys that must exist when observation payloads are mappings.
    required_outcome_fields : tuple[str, ...], optional
        Outcome fields that must exist as attributes or mapping keys.

    Notes
    -----
    Requirements are declarative metadata. Validation is handled by
    :mod:`comp_model_v2.inference.compatibility`.
    """

    required_observation_fields: tuple[str, ...] = ()
    required_outcome_fields: tuple[str, ...] = ()
