from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

from ..data.types import StudyData
from ..interfaces.estimator import Estimator, FitResult
from ..interfaces.model import ComputationalModel
from ..likelihood.replay import loglike_subject
from ..params.bounds import ParameterBoundsSpace
from ..errors import CompatibilityError
from ..utility import _as_scipy_bounds

@dataclass(slots=True)
class SubjectwiseMLEEstimator(Estimator):
    """
    Generic subject-wise MLE with box constraints (no transforms).

    - fits each subject independently
    - uses shared replay likelihood
    - optimizes directly in parameter space with bounds
    """
    space: ParameterBoundsSpace

    # Multi-start
    n_starts: int = 20

    # Local optimizer options
    method: str = "L-BFGS-B"
    maxiter: int = 300

    def supports(self, study: StudyData, model: Any) -> bool:
        return isinstance(model, ComputationalModel)

    def fit(self, *, study: StudyData, model: Any, rng: np.random.Generator) -> FitResult:
        if not self.supports(study, model):
            raise CompatibilityError("Given `model` is not compatible with `SubjectwiseMLEEstimator`")

        bounds = _as_scipy_bounds(self.space)

        subj_hats: dict[str, dict[str, float]] = {}
        total_ll = 0.0
        diags: dict[str, Any] = {
            "optimizer": self.method,
            "n_starts": self.n_starts,
        }

        for subj in study.subjects:
            # objective: minimize negative log-likelihood
            def nll(x: np.ndarray) -> float:
                params = self.space.to_params(x)
                ll = loglike_subject(subject=subj, model=model, params=params)
                return float(-ll)

            best_x: np.ndarray | None = None
            best_fun: float = float("inf")
            best_res = None

            # Multi-start local optimization
            for _ in range(self.n_starts):
                x0 = self.space.sample_init(rng)
                res = minimize(
                    fun=nll,
                    x0=x0,
                    method=self.method,
                    bounds=bounds,
                    options={"maxiter": self.maxiter},
                )
                if float(res.fun) < best_fun:
                    best_fun = float(res.fun)
                    best_x = np.array(res.x, dtype=float)
                    best_res = res

            assert best_x is not None
            params_hat = self.space.to_params(best_x)
            subj_hats[subj.subject_id] = params_hat
            total_ll += float(-best_fun)

            # per-subject diagnostics (lightweight)
            diags[f"subj_{subj.subject_id}"] = {
                "fun": float(best_fun),
                "success": bool(getattr(best_res, "success", True)),
                "message": str(getattr(best_res, "message", "")),
                "nit": int(getattr(best_res, "nit", -1)),
            }

        return FitResult(
            subject_hats=subj_hats,
            value=float(total_ll),
            success=True,
            message="OK",
            diagnostics=diags,
        )
