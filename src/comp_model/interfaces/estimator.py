from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..data.types import StudyData


@dataclass(frozen=True, slots=True)
class FitResult:
    params_hat: Mapping[str, float] | None = None
    population_hat: Mapping[str, float] | None = None
    subject_hats: Mapping[str, Mapping[str, float]] | None = None
    value: float | None = None
    success: bool = True
    message: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)


class Estimator(ABC):
    def supports(self, study: StudyData, model: Any) -> bool:
        return True

    @abstractmethod
    def fit(
        self,
        *,
        study: StudyData,
        model: Any,
        rng: np.random.Generator,
    ) -> FitResult:
        ...
