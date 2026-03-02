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
        # Hierarchical posterior-style result:
        # ``result.map_candidate.block_params`` with per-block parameters.
        if hasattr(map_candidate, "block_params"):
            params = _hierarchical_params_from_map_candidate(map_candidate)
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


def _hierarchical_params_from_map_candidate(map_candidate: Any) -> dict[str, float]:
    """Extract one representative parameter mapping from hierarchical MAP candidate.

    Parameters
    ----------
    map_candidate : Any
        MAP candidate object exposing ``block_params`` and optionally
        ``parameter_names``.

    Returns
    -------
    dict[str, float]
        Mean parameter values across block-specific maps.
    """

    block_params_raw = getattr(map_candidate, "block_params", None)
    if not isinstance(block_params_raw, (tuple, list)):
        raise TypeError("fit_result.map_candidate.block_params must be a sequence")
    if len(block_params_raw) == 0:
        return {}

    block_params: list[dict[str, float]] = []
    for index, block in enumerate(block_params_raw):
        block_params.append(
            _coerce_float_mapping(
                block,
                field_name=f"fit_result.map_candidate.block_params[{index}]",
            )
        )

    names_raw = getattr(map_candidate, "parameter_names", None)
    if isinstance(names_raw, (tuple, list)) and len(names_raw) > 0:
        names = tuple(str(name) for name in names_raw)
    else:
        names = tuple(block_params[0].keys())

    out: dict[str, float] = {}
    for name in names:
        values = [float(block[name]) for block in block_params if name in block]
        if not values:
            continue
        out[name] = float(sum(values) / len(values))
    return out


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
