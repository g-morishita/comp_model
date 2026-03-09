"""Helpers for reading best points from MLE fit results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class BestFitSummary:
    """Best-point summary extracted from an MLE fit result."""

    params: dict[str, float]
    log_likelihood: float
    raw_result: Any


def extract_best_fit_summary(fit_result: Any) -> BestFitSummary:
    """Extract best-parameter summary from an MLE fit result."""

    best = getattr(fit_result, "best", None)
    if best is None:
        raise TypeError(
            "unsupported fit result type; expected an object with "
            "`.best.params` and `.best.log_likelihood`"
        )

    return BestFitSummary(
        params=_coerce_float_mapping(
            getattr(best, "params", None),
            field_name="fit_result.best.params",
        ),
        log_likelihood=_coerce_float(
            getattr(best, "log_likelihood", None),
            field_name="fit_result.best.log_likelihood",
        ),
        raw_result=fit_result,
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
