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

from ..likelihood.replay import loglike_subject


@dataclass(slots=True)
class BoxMLESubjectwiseEstimator(Estimator):
    """
    Subject-wise MLE with multi-start box-constrained optimization.

    If `space` is None, it is derived from `model.param_schema`.
    """

    model: ComputationalModel
    space: ParameterBoundsSpace | None = None

    n_starts: int = 20
    method: str = "L-BFGS-B"
    maxiter: int = 300

    # safety: whether to validate bounds when setting params (usually False in MLE loop for speed)
    validate_bounds_on_set: bool = False

    def __post_init__(self) -> None:
        self._ensure_space()
        self.assert_param_space()

    def _ensure_space(self) -> None:
        if self.space is not None:
            return
        # Derived from schema (single source of truth)
        self.space = self.model.param_schema.bounds_space(
            names=self.model.param_schema.names,
            require_bounds=True,
        )

    def assert_param_space(self) -> None:
        assert self.space is not None
        model_names = tuple(self.model.param_schema.names)
        space_names = tuple(self.space.names)
        if set(model_names) != set(space_names):
            raise ValueError(
                "Model/space parameter mismatch: "
                f"model={model_names}, space={space_names}"
            )

    def supports(self, study: StudyData) -> bool:
        for subj in study.subjects:
            for blk in subj.blocks:
                if not self.model.supports(blk.task_spec):
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
            "optimizer": self.method,
            "n_starts": self.n_starts,
            "maxiter": self.maxiter,
            "space_names": list(self.space.names),
        }

        for subj in study.subjects:

            def nll(x: np.ndarray) -> float:
                params = self.space.to_params(x)

                # For safety, you *can* validate schema here, but it’s slower.
                # Most of the time, bounds already ensure validity.
                if self.validate_bounds_on_set:
                    self.model.set_params(params, strict=True, check_bounds=True)
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
    Subject-wise MLE with *transformed/unconstrained* optimization.

    - Optimizes z in R^d, then maps to constrained params using model.param_schema.params_from_z(z).
    - This is often more stable than box constraints for RL models.
    """

    model: ComputationalModel

    # multi-start
    n_starts: int = 20

    # optimizer
    method: str = "L-BFGS-B"
    maxiter: int = 300

    # init in z-space: Gaussian around default z
    z_init_scale: float = 1.0
    

    def supports(self, study: StudyData) -> bool:
        for subj in study.subjects:
            for blk in subj.blocks:
                if not self.model.supports(blk.task_spec):
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

            # Start at default z
            z0 = schema.default_z()
            res = minimize(
                fun=nll_z,
                x0=z0,
                method=self.method,
                options={"maxiter": self.maxiter},
            )
            best_fun = float(res.fun)
            best_z = np.array(res.x, dtype=float)
            best_res = res

            # Additional random starts
            for _ in range(max(0, self.n_starts - 1)):
                z0 = schema.sample_z_init(rng, center="default", scale=float(self.z_init_scale))
                res = minimize(
                    fun=nll_z,
                    x0=z0,
                    method=self.method,
                    options={"maxiter": self.maxiter},
                )
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