from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

Json = dict[str, Any]


@dataclass(frozen=True, slots=True)
class RecoveryTruth:
    """What was used to generate data."""
    population: Mapping[str, float] | None
    subjects: Mapping[str, Mapping[str, float]]  # subject_id -> params


@dataclass(frozen=True, slots=True)
class RecoveryEstimate:
    """What the estimator returned."""
    population: Mapping[str, float] | None
    subjects: Mapping[str, Mapping[str, float]] | None  # may be None if pooled


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    run_id: str
    seed: int

    generating_model: str
    fitted_model: str
    task: str

    truth: RecoveryTruth
    estimate: RecoveryEstimate

    success: bool
    message: str = ""
    diagnostics: Json = field(default_factory=dict)
    metadata: Json = field(default_factory=dict)
