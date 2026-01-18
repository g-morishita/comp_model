from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize

from ..data.types import StudyData
from ..interfaces.estimator import Estimator, FitResult
from ..likelihood.replay import loglike_subject
from ..models.vs.vs import VS
from ..params.bounds import ParameterBoundsSpace


def _as_scipy_bounds(space: ParameterBoundsSpace) -> list[tuple[float, float]]:
    return [(space.bounds[name].lo, space.bounds[name].hi) for name in space.names]


@dataclass(slots=True)
class VSMLEEstimator(Estimator):
    """
    SciPy MLE for VS (K-armed), optimizing directly with box constraints.

    Strategy (per subject):
      - generate n_starts initial points uniformly within bounds
      - for each start: run L-BFGS-B on negative log-likelihood
      - keep best solution

    Optional:
      - use differential_evolution once per subject to get a strong global init.

    Notes:
      - If you see alpha hitting 0/1, tighten bounds (e.g., eps=1e-3) in vs_bounds_space.
      - Works with any number of arms; VS reads spec.n_actions in replay.
    """
    space: ParameterBoundsSpace

    # Multi-start
    n_starts: int = 20

    # Local optimizer options
    method: str = "L-BFGS-B"
    maxiter: int = 300

    def supports(self, study: StudyData, model: Any) -> bool:
        return isinstance(model, VS)

    def fit(self, *, study: StudyData, model: Any, rng: np.random.Generator) -> FitResult:
        if not isinstance(model, VS):
            return FitResult(success=False, message="VSMLEEstimatorSciPy expects a VS model instance.")

        bounds = _as_scipy_bounds(self.space)
        names = tuple(self.space.names)

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
