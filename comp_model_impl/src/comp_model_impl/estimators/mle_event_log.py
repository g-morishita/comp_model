"""Maximum-likelihood estimation using event-log replay.

This module provides subject-wise MLE estimators that fit model parameters by
replaying an event log attached to each block. The event stream defines
*exactly* when model reset, social observation, choice, and outcome updates
occur.

Notes
-----
Event-log replay is the sole likelihood source. If the event log is missing or
incompatible with the model, fitting will fail.

Examples
--------
Fit a model with box-constrained parameters:

>>> import numpy as np
>>> from comp_model_impl.estimators.mle_event_log import BoxMLESubjectwiseEstimator
>>> from comp_model_impl.models import QRL
>>> estimator = BoxMLESubjectwiseEstimator(model=QRL(), n_starts=5)
>>> # estimator.fit(study=study, rng=np.random.default_rng(0))  # doctest: +SKIP
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Mapping, Sequence

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


def _resolve_fixed_params(
    *,
    model: ComputationalModel,
    fixed_params: Mapping[str, float] | Sequence[str] | None,
) -> dict[str, float]:
    """Resolve fixed-parameter values from a mapping or list of names.

    If ``fixed_params`` is a mapping, its values are used directly (after
    validation). If it is a sequence of names, values are taken from the
    current model parameters.
    """
    if fixed_params is None:
        return {}

    if isinstance(fixed_params, Mapping):
        fixed = {str(k): float(v) for k, v in fixed_params.items()}
    else:
        if isinstance(fixed_params, str):
            names = [fixed_params]
        else:
            names = [str(n) for n in fixed_params]
        base_params = model.get_params()
        missing = [n for n in names if n not in base_params]
        if missing:
            raise ValueError(f"Unknown fixed parameter(s): {missing}")
        fixed = {n: float(base_params[n]) for n in names}

    # Validate names and bounds.
    return model.param_schema.validate(fixed, strict=True, check_bounds=True)


@dataclass(slots=True)
class BoxMLESubjectwiseEstimator(Estimator):
    """Subject-wise MLE with multi-start box-constrained optimization.

    Parameters
    ----------
    model : comp_model_core.interfaces.model.ComputationalModel
        Model to be fit. Its :attr:`param_schema` defines parameter names and
        bounds.
    space : comp_model_core.params.ParameterBoundsSpace or None, optional
        Parameter space to optimize in. If ``None``, a bounds space is derived
        from the model schema.
    n_starts : int, optional
        Number of random restarts for optimization.
    method : str, optional
        SciPy optimizer name (default: ``"L-BFGS-B"``).
    maxiter : int, optional
        Maximum number of optimizer iterations.
    validate_bounds_on_set : bool, optional
        If ``True``, validates parameters against schema bounds on each
        likelihood call (slower but stricter).

    Notes
    -----
    Uses event-log likelihood replay (``BLOCK_START`` controls reset timing).
    """
    model: ComputationalModel
    space: ParameterBoundsSpace | None = None

    n_starts: int = 20
    method: str = "L-BFGS-B"
    maxiter: int = 300

    validate_bounds_on_set: bool = False

    def __post_init__(self) -> None:
        """Initialize defaults and validate parameter-space consistency.

        Notes
        -----
        If ``space`` is not provided, a bounds space is derived from the model
        schema. The derived space must match the model parameter names exactly.
        """
        if self.space is None:
            self.space = self.model.param_schema.bounds_space(
                names=self.model.param_schema.names,
                require_bounds=True,
            )
        # Ensure the optimization space matches the model schema exactly.
        model_names = set(self.model.param_schema.names)
        space_names = set(self.space.names)
        if model_names != space_names:
            raise ValueError(f"Model/space parameter mismatch: model={model_names}, space={space_names}")

    def supports(self, study: StudyData) -> bool:
        """Check whether the estimator can fit the provided study.

        Parameters
        ----------
        study : comp_model_core.data.types.StudyData
            Study data containing subjects and blocks with event logs.

        Returns
        -------
        bool
            ``True`` if all blocks have compatible environment specs for the
            model; ``False`` otherwise.
        """
        for subj in study.subjects:
            for blk in subj.blocks:
                if blk.env_spec is None or not self.model.supports(blk.env_spec):
                    return False
        return True

    def fit(
        self,
        *,
        study: StudyData,
        rng: np.random.Generator,
        fixed_params: Mapping[str, float] | Sequence[str] | None = None,
    ) -> FitResult:
        """Fit parameters by minimizing negative log-likelihood per subject.

        Parameters
        ----------
        study : comp_model_core.data.types.StudyData
            Study data to fit, including event logs per block.
        rng : numpy.random.Generator
            Random number generator used for initialization.
        fixed_params : Mapping[str, float] or Sequence[str] or None, optional
            Parameters to hold fixed. If a mapping, values are used directly.
            If a sequence of names, values are taken from the current model.

        Returns
        -------
        comp_model_core.interfaces.estimator.FitResult
            Estimated parameters and diagnostics.
        """
        if not self.supports(study):
            raise CompatibilityError("Given data is not compatible with the computational model")

        assert self.space is not None
        fixed = _resolve_fixed_params(model=self.model, fixed_params=fixed_params)

        if fixed:
            free_names = [n for n in self.space.names if n not in fixed]
            free_space = ParameterBoundsSpace(
                names=tuple(free_names),
                bounds={n: self.space.bounds[n] for n in free_names},
            )
        else:
            free_space = self.space
            free_names = list(self.space.names)

        bounds = _as_scipy_bounds(free_space)

        subj_hats: dict[str, dict[str, float]] = {}
        total_ll = 0.0

        diags: dict[str, Any] = {
            "estimator": self.__class__.__name__,
            "optimizer": self.method,
            "n_starts": self.n_starts,
            "maxiter": self.maxiter,
            "space_names": list(self.space.names),
            "fixed_params": dict(fixed),
            "free_names": list(free_names),
        }

        for subj in study.subjects:
            if fixed:
                def nll(x: np.ndarray) -> float:
                    # Map free vector to named parameters and compute NLL.
                    params = dict(fixed)
                    if free_space.dim > 0:
                        params.update(free_space.to_params(x))
                    if self.validate_bounds_on_set:
                        params = self.model.param_schema.validate(params, strict=True, check_bounds=True)
                    ll = loglike_subject(subject=subj, model=self.model, params=params)
                    return float(-ll)
            else:
                def nll(x: np.ndarray) -> float:
                    # Map vector to named parameters and compute NLL.
                    params = free_space.to_params(x)
                    if self.validate_bounds_on_set:
                        params = self.model.param_schema.validate(params, strict=True, check_bounds=True)
                    ll = loglike_subject(subject=subj, model=self.model, params=params)
                    return float(-ll)

            best_x: np.ndarray | None = None
            best_fun: float = float("inf")
            best_res = None

            if free_space.dim == 0:
                best_x = np.zeros((0,), dtype=float)
                best_fun = float(nll(best_x))
            else:
                for _ in range(self.n_starts):
                    # Multi-start improves robustness to local optima.
                    x0 = free_space.sample_init(rng)
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
            params_hat = dict(fixed)
            if free_space.dim > 0:
                params_hat.update(free_space.to_params(best_x))

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
    """Subject-wise MLE with unconstrained optimization in z-space.

    Parameters
    ----------
    model : comp_model_core.interfaces.model.ComputationalModel
        Model to be fit. Its schema defines the z-transform and default
        initialization.
    n_starts : int, optional
        Number of random restarts (in addition to the default start).
    method : str, optional
        SciPy optimizer name (default: ``"L-BFGS-B"``).
    maxiter : int, optional
        Maximum number of optimizer iterations.
    z_init_scale : float, optional
        Scale factor for random z initializations around the default.

    Notes
    -----
    Uses event-log likelihood replay as the objective function.
    """
    model: ComputationalModel

    n_starts: int = 20
    method: str = "L-BFGS-B"
    maxiter: int = 300
    z_init_scale: float = 1.0

    def supports(self, study: StudyData) -> bool:
        """Check whether the estimator can fit the provided study.

        Parameters
        ----------
        study : comp_model_core.data.types.StudyData
            Study data containing subjects and blocks with event logs.

        Returns
        -------
        bool
            ``True`` if all blocks have compatible environment specs for the
            model; ``False`` otherwise.
        """
        for subj in study.subjects:
            for blk in subj.blocks:
                if blk.env_spec is None or not self.model.supports(blk.env_spec):
                    return False
        return True

    def fit(
        self,
        *,
        study: StudyData,
        rng: np.random.Generator,
        fixed_params: Mapping[str, float] | Sequence[str] | None = None,
    ) -> FitResult:
        """Fit parameters by optimizing in z-space per subject.

        Parameters
        ----------
        study : comp_model_core.data.types.StudyData
            Study data to fit, including event logs per block.
        rng : numpy.random.Generator
            Random number generator used for initialization.
        fixed_params : Mapping[str, float] or Sequence[str] or None, optional
            Parameters to hold fixed. If a mapping, values are used directly.
            If a sequence of names, values are taken from the current model.

        Returns
        -------
        comp_model_core.interfaces.estimator.FitResult
            Estimated parameters and diagnostics.
        """
        if not self.supports(study):
            raise CompatibilityError("Given data is not compatible with the computational model")

        schema = self.model.param_schema
        fixed = _resolve_fixed_params(model=self.model, fixed_params=fixed_params)
        names = schema.names
        dim = len(names)
        if fixed:
            free_names = [n for n in names if n not in fixed]
            free_idx = np.array([names.index(n) for n in free_names], dtype=int)
        else:
            free_names = list(names)
            free_idx = np.arange(dim, dtype=int)
        free_dim = int(len(free_idx))

        subj_hats: dict[str, dict[str, float]] = {}
        total_ll = 0.0

        diags: dict[str, Any] = {
            "estimator": self.__class__.__name__,
            "optimizer": self.method,
            "n_starts": self.n_starts,
            "maxiter": self.maxiter,
            "param_names": list(names),
            "z_init_scale": float(self.z_init_scale),
            "fixed_params": dict(fixed),
            "free_names": list(free_names),
        }

        for subj in study.subjects:
            z_base = schema.default_z()

            def nll_z(z: np.ndarray) -> float:
                z = np.asarray(z, dtype=float)
                if z.shape != (free_dim,):
                    raise ValueError(f"Expected z.shape == ({free_dim},), got {z.shape}")
                z_full = z_base.copy()
                if free_dim > 0:
                    z_full[free_idx] = z
                # Transform from unconstrained z to bounded parameters.
                params = schema.params_from_z(z_full)
                if fixed:
                    params.update(fixed)
                ll = loglike_subject(subject=subj, model=self.model, params=params)
                return float(-ll)

            best_fun = float("inf")
            best_z: np.ndarray | None = None
            best_res = None

            if free_dim == 0:
                best_z = np.zeros((0,), dtype=float)
                best_fun = float(nll_z(best_z))
            else:
                # Start at the schema default z for a stable baseline.
                z0_full = schema.default_z()
                z0 = z0_full[free_idx]
                res = minimize(fun=nll_z, x0=z0, method=self.method, options={"maxiter": self.maxiter})
                best_fun = float(res.fun)
                best_z = np.array(res.x, dtype=float)
                best_res = res

                # Random restarts around the default improve robustness.
                for _ in range(max(0, self.n_starts - 1)):
                    z0_full = schema.sample_z_init(rng, center="default", scale=float(self.z_init_scale))
                    z0 = z0_full[free_idx]
                    res = minimize(fun=nll_z, x0=z0, method=self.method, options={"maxiter": self.maxiter})
                    if float(res.fun) < best_fun:
                        best_fun = float(res.fun)
                        best_z = np.array(res.x, dtype=float)
                        best_res = res

            assert best_z is not None
            z_full = z_base.copy()
            if free_dim > 0:
                z_full[free_idx] = best_z
            params_hat = schema.params_from_z(z_full)
            if fixed:
                params_hat.update(fixed)

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
