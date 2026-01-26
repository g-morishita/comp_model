from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

from comp_model_core.data.types import StudyData
from comp_model_core.errors import CompatibilityError
from comp_model_core.interfaces.estimator import Estimator, FitResult
from comp_model_core.interfaces.model import ComputationalModel
from comp_model_core.params import ParameterBoundsSpace
from comp_model_core.utility import _as_scipy_bounds

# CHANGED: event-log replay is the only likelihood source
from comp_model_impl.likelihood.event_log_replay import loglike_subject


@dataclass(slots=True)
class BoxMLESubjectwiseEstimator(Estimator):
    """
    Subject-wise MLE with multi-start box-constrained optimization.
    Uses event-log likelihood replay (BLOCK_START controls reset timing).
    """
    model: ComputationalModel
    space: ParameterBoundsSpace | None = None

    n_starts: int = 20
    method: str = "L-BFGS-B"
    maxiter: int = 300

    validate_bounds_on_set: bool = False

    def __post_init__(self) -> None:
        if self.space is None:
            self.space = self.model.param_schema.bounds_space(
                names=self.model.param_schema.names,
                require_bounds=True,
            )
        # sanity
        model_names = set(self.model.param_schema.names)
        space_names = set(self.space.names)
        if model_names != space_names:
            raise ValueError(f"Model/space parameter mismatch: model={model_names}, space={space_names}")

    def supports(self, study: StudyData) -> bool:
        for subj in study.subjects:
            for blk in subj.blocks:
                if blk.env_spec is None or not self.model.supports(blk.env_spec):
                    return False
        return True

    def fit(self, *, study: StudyData, rng: np.random.Generator) -> FitResult:
        if not self.supports(study):
            raise CompatibilityError("Given data is not compatible with the computational model")

        assert self.space is not None
        bounds = _as_scipy_bounds(self.space)

        subj_hats: dict[str, dict[str, float]] = {}
        total_ll = 0.0

        diags: dict[str, Any] = {
            "estimator": self.__class__.__name__,
            "optimizer": self.method,
            "n_starts": self.n_starts,
            "maxiter": self.maxiter,
            "space_names": list(self.space.names),
        }

        for subj in study.subjects:

            def nll(x: np.ndarray) -> float:
                params = self.space.to_params(x)
                if self.validate_bounds_on_set:
                    params = self.model.param_schema.validate(params, strict=True, check_bounds=True)
                ll = loglike_subject(subject=subj, model=self.model, params=params)
                return float(-ll)

            best_x: np.ndarray | None = None
            best_fun: float = float("inf")
            best_res = None

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


@dataclass(slots=True)
class TransformedMLESubjectwiseEstimator(Estimator):
    """
    Subject-wise MLE with unconstrained optimization in z-space.
    Uses event-log likelihood replay.
    """
    model: ComputationalModel

    n_starts: int = 20
    method: str = "L-BFGS-B"
    maxiter: int = 300
    z_init_scale: float = 1.0

    def supports(self, study: StudyData) -> bool:
        for subj in study.subjects:
            for blk in subj.blocks:
                if blk.env_spec is None or not self.model.supports(blk.env_spec):
                    return False
        return True

    def fit(self, *, study: StudyData, rng: np.random.Generator) -> FitResult:
        if not self.supports(study):
            raise CompatibilityError("Given data is not compatible with the computational model")

        schema = self.model.param_schema
        names = schema.names
        dim = len(names)

        subj_hats: dict[str, dict[str, float]] = {}
        total_ll = 0.0

        diags: dict[str, Any] = {
            "estimator": self.__class__.__name__,
            "optimizer": self.method,
            "n_starts": self.n_starts,
            "maxiter": self.maxiter,
            "param_names": list(names),
            "z_init_scale": float(self.z_init_scale),
        }

        for subj in study.subjects:

            def nll_z(z: np.ndarray) -> float:
                z = np.asarray(z, dtype=float)
                if z.shape != (dim,):
                    raise ValueError(f"Expected z.shape == ({dim},), got {z.shape}")
                params = schema.params_from_z(z)
                ll = loglike_subject(subject=subj, model=self.model, params=params)
                return float(-ll)

            best_fun = float("inf")
            best_z: np.ndarray | None = None
            best_res = None

            # start at default z
            z0 = schema.default_z()
            res = minimize(fun=nll_z, x0=z0, method=self.method, options={"maxiter": self.maxiter})
            best_fun = float(res.fun)
            best_z = np.array(res.x, dtype=float)
            best_res = res

            # random starts
            for _ in range(max(0, self.n_starts - 1)):
                z0 = schema.sample_z_init(rng, center="default", scale=float(self.z_init_scale))
                res = minimize(fun=nll_z, x0=z0, method=self.method, options={"maxiter": self.maxiter})
                if float(res.fun) < best_fun:
                    best_fun = float(res.fun)
                    best_z = np.array(res.x, dtype=float)
                    best_res = res

            assert best_z is not None
            params_hat = schema.params_from_z(best_z)

            subj_hats[subj.subject_id] = params_hat
            total_ll += float(-best_fun)

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
