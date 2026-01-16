from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..data.types import StudyData
from ..interfaces.estimator import Estimator, FitResult
from ..interfaces.model import ComputationalModel
from ..likelihood.replay import loglike_subject
from ..params.bounds import ParameterBoundsSpace
from .box_opt import RandomRestartCoordinateAscentBox


@dataclass(slots=True)
class SubjectwiseBoxMLEEstimator(Estimator):
    """
    Generic subject-wise MLE with box constraints:
      - fits each subject independently
      - uses shared replay likelihood
      - optimizes parameters directly with bounds (no transforms)
    """
    space: ParameterBoundsSpace
    optimizer: RandomRestartCoordinateAscentBox = RandomRestartCoordinateAscentBox()

    def supports(self, study: StudyData, model: Any) -> bool:
        return isinstance(model, ComputationalModel)

    def fit(self, *, study: StudyData, model: Any, rng: np.random.Generator) -> FitResult:
        assert isinstance(model, ComputationalModel)

        subj_hats: dict[str, dict[str, float]] = {}
        total_ll = 0.0

        for subj in study.subjects:
            def obj(x: np.ndarray) -> float:
                params = self.space.to_params(x)
                return loglike_subject(study=study, subject=subj, model=model, params=params)

            x_hat, ll_hat = self.optimizer.maximize(obj, rng=rng, space=self.space)
            params_hat = self.space.to_params(x_hat)
            subj_hats[subj.subject_id] = params_hat
            total_ll += float(ll_hat)

        return FitResult(
            subject_hats=subj_hats,
            value=float(total_ll),
            success=True,
            message="OK",
            diagnostics={},
        )
