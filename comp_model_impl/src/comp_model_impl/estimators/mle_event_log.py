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
from scipy.stats import norm

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


def _normal_z_for_ci(ci_level: float) -> float:
    """Return the normal critical value for a two-sided CI level."""
    q = float(ci_level)
    if not (0.0 < q < 1.0):
        raise ValueError(f"uncertainty_ci must be in (0, 1), got {ci_level!r}")
    return float(norm.ppf(0.5 + 0.5 * q))


def _cov_from_opt_result(opt_res: Any, *, dim: int) -> np.ndarray | None:
    """Extract approximate covariance from a SciPy optimization result."""
    if int(dim) <= 0 or opt_res is None:
        return None
    h_inv = getattr(opt_res, "hess_inv", None)
    if h_inv is None:
        return None
    try:
        if hasattr(h_inv, "todense"):
            cov = np.asarray(h_inv.todense(), dtype=float)
        else:
            cov = np.asarray(h_inv, dtype=float)
    except Exception:
        return None
    if cov.shape != (int(dim), int(dim)):
        return None
    if np.any(~np.isfinite(cov)):
        return None
    # Symmetrize for numerical stability.
    cov = 0.5 * (cov + cov.T)
    return cov


def _uncertainty_from_cov(
    *,
    params_hat: Mapping[str, float],
    param_names: Sequence[str],
    cov: np.ndarray,
    ci_level: float,
) -> dict[str, dict[str, float]]:
    """Build per-parameter uncertainty summaries from covariance."""
    zcrit = _normal_z_for_ci(ci_level)
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(param_names):
        name = str(name)
        var = float(cov[i, i])
        se = float(np.sqrt(var)) if np.isfinite(var) and var >= 0.0 else float("nan")
        hat = float(params_hat[name])
        if np.isfinite(se):
            ci_lo = float(hat - zcrit * se)
            ci_hi = float(hat + zcrit * se)
        else:
            ci_lo = float("nan")
            ci_hi = float("nan")
        out[name] = {
            "hat": hat,
            "se": se,
            "var": var,
            "ci_level": float(ci_level),
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
        }
    return out


def _jacobian_params_wrt_free_z(
    *,
    schema: Any,
    z_base: np.ndarray,
    z_hat_free: np.ndarray,
    free_idx: np.ndarray,
    free_names: Sequence[str],
    fixed: Mapping[str, float],
    step: float,
) -> np.ndarray:
    """Numerically approximate Jacobian of free params wrt free z."""
    free_dim = int(len(free_idx))
    if free_dim == 0:
        return np.zeros((0, 0), dtype=float)

    h_scale = float(step)
    if h_scale <= 0:
        raise ValueError(f"uncertainty_fd_step must be > 0, got {step!r}")

    def _params_from_free_z(zf: np.ndarray) -> np.ndarray:
        z_full = z_base.copy()
        z_full[free_idx] = zf
        p = dict(schema.params_from_z(z_full))
        if fixed:
            p.update(fixed)
        return np.asarray([float(p[n]) for n in free_names], dtype=float)

    f0 = _params_from_free_z(z_hat_free)
    J = np.zeros((len(free_names), free_dim), dtype=float)
    for j in range(free_dim):
        h = float(h_scale * max(1.0, abs(float(z_hat_free[j]))))
        zp = np.array(z_hat_free, dtype=float)
        zm = np.array(z_hat_free, dtype=float)
        zp[j] += h
        zm[j] -= h
        fp = _params_from_free_z(zp)
        fm = _params_from_free_z(zm)
        deriv = (fp - fm) / (2.0 * h)
        J[:, j] = deriv
    # If central differences produce NaNs, fallback to zeros for stability.
    if np.any(~np.isfinite(J)):
        return np.zeros_like(J)
    # Keep f0 referenced to make intent explicit for potential future extensions.
    _ = f0
    return J


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
    return_uncertainty : bool, optional
        If ``True``, include approximate parameter uncertainty from the
        optimizer inverse-Hessian in diagnostics.
    uncertainty_ci : float, optional
        Two-sided confidence interval level for reported bounds.

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
    return_uncertainty: bool = False
    uncertainty_ci: float = 0.95

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
            "return_uncertainty": bool(self.return_uncertainty),
            "uncertainty_ci": float(self.uncertainty_ci),
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
            if self.return_uncertainty and free_space.dim > 0:
                cov_free = _cov_from_opt_result(best_res, dim=free_space.dim)
                if cov_free is not None:
                    unc = _uncertainty_from_cov(
                        params_hat=params_hat,
                        param_names=free_names,
                        cov=cov_free,
                        ci_level=float(self.uncertainty_ci),
                    )
                    diags[f"subj_{subj.subject_id}"]["uncertainty"] = {
                        "method": "hessian_inv",
                        "params": unc,
                    }
                else:
                    diags[f"subj_{subj.subject_id}"]["uncertainty"] = {
                        "method": "hessian_inv",
                        "params": {},
                        "warning": "inverse-Hessian unavailable from optimizer result",
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
    return_uncertainty : bool, optional
        If ``True``, include approximate parameter uncertainty via inverse
        Hessian in z-space and a delta-method transform to parameter space.
    uncertainty_ci : float, optional
        Two-sided confidence interval level for reported bounds.
    uncertainty_fd_step : float, optional
        Relative finite-difference step used for the delta-method Jacobian.

    Notes
    -----
    Uses event-log likelihood replay as the objective function.
    """
    model: ComputationalModel

    n_starts: int = 20
    method: str = "L-BFGS-B"
    maxiter: int = 300
    z_init_scale: float = 1.0
    return_uncertainty: bool = False
    uncertainty_ci: float = 0.95
    uncertainty_fd_step: float = 1e-5

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
            "return_uncertainty": bool(self.return_uncertainty),
            "uncertainty_ci": float(self.uncertainty_ci),
            "uncertainty_fd_step": float(self.uncertainty_fd_step),
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
            if self.return_uncertainty and free_dim > 0:
                cov_z = _cov_from_opt_result(best_res, dim=free_dim)
                if cov_z is not None:
                    J = _jacobian_params_wrt_free_z(
                        schema=schema,
                        z_base=z_base,
                        z_hat_free=best_z,
                        free_idx=free_idx,
                        free_names=free_names,
                        fixed=fixed,
                        step=float(self.uncertainty_fd_step),
                    )
                    cov_p = J @ cov_z @ J.T
                    unc = _uncertainty_from_cov(
                        params_hat=params_hat,
                        param_names=free_names,
                        cov=cov_p,
                        ci_level=float(self.uncertainty_ci),
                    )
                    diags[f"subj_{subj.subject_id}"]["uncertainty"] = {
                        "method": "delta_method_from_z_hessian_inv",
                        "params": unc,
                    }
                else:
                    diags[f"subj_{subj.subject_id}"]["uncertainty"] = {
                        "method": "delta_method_from_z_hessian_inv",
                        "params": {},
                        "warning": "inverse-Hessian unavailable from optimizer result",
                    }

        return FitResult(
            subject_hats=subj_hats,
            value=float(total_ll),
            success=True,
            message="OK",
            diagnostics=diags,
        )
