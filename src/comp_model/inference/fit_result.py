"""Helpers for reading best points from inference fit results.

The library currently supports multiple fit-result families (for example MLE
and MAP). This module provides a small compatibility layer so downstream
workflows can consume either result shape via one interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class BestFitSummary:
    """Best-point summary extracted from an inference fit result.

    Parameters
    ----------
    params : dict[str, float]
        Best parameter mapping.
    log_likelihood : float
        Log-likelihood at the best point.
    log_posterior : float | None
        Log-posterior at the best point when available.
    raw_result : Any
        Original fit result object.
    """

    params: dict[str, float]
    log_likelihood: float
    log_posterior: float | None
    raw_result: Any


def extract_best_fit_summary(fit_result: Any) -> BestFitSummary:
    """Extract a unified best-point summary from a fit result object.

    Parameters
    ----------
    fit_result : Any
        Fit result object from inference APIs.

    Returns
    -------
    BestFitSummary
        Unified best-point representation.

    Raises
    ------
    TypeError
        If ``fit_result`` does not expose a supported best-candidate shape.
    """

    # MLE-style result: ``result.best.params`` and ``result.best.log_likelihood``.
    best = getattr(fit_result, "best", None)
    if best is not None:
        params = _coerce_float_mapping(getattr(best, "params", None), field_name="fit_result.best.params")
        log_likelihood = _coerce_float(getattr(best, "log_likelihood", None), field_name="fit_result.best.log_likelihood")
        return BestFitSummary(
            params=params,
            log_likelihood=log_likelihood,
            log_posterior=None,
            raw_result=fit_result,
        )

    # MAP-style result: ``result.map_candidate.params`` etc.
    map_candidate = getattr(fit_result, "map_candidate", None)
    if map_candidate is not None:
        params = _coerce_float_mapping(
            getattr(map_candidate, "params", None),
            field_name="fit_result.map_candidate.params",
        )
        log_likelihood = _coerce_float(
            getattr(map_candidate, "log_likelihood", None),
            field_name="fit_result.map_candidate.log_likelihood",
        )
        log_posterior = _coerce_float(
            getattr(map_candidate, "log_posterior", None),
            field_name="fit_result.map_candidate.log_posterior",
        )
        return BestFitSummary(
            params=params,
            log_likelihood=log_likelihood,
            log_posterior=log_posterior,
            raw_result=fit_result,
        )

    raise TypeError(
        "unsupported fit result type; expected an object with either "
        "`.best` (MLE-style) or `.map_candidate` (MAP-style)"
    )


def _coerce_float_mapping(raw: Any, *, field_name: str) -> dict[str, float]:
    """Coerce mapping-like value to ``dict[str, float]`` with validation."""

    if not isinstance(raw, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return {str(key): float(value) for key, value in raw.items()}


def _coerce_float(raw: Any, *, field_name: str) -> float:
    """Coerce a scalar to float with clearer errors."""

    if raw is None:
        raise TypeError(f"{field_name} is required")
    return float(raw)


__all__ = ["BestFitSummary", "extract_best_fit_summary"]
