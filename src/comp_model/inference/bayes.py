"""Bayesian inference primitives and MAP estimators.

This module introduces a backend-agnostic Bayesian interface centered on
log-prior programs and MAP estimation. It is designed as the first step toward
full Bayesian workflows (including hierarchical variants) while reusing the
canonical replay likelihood stack.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import lgamma, log, log1p, pi
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from comp_model.core.contracts import AgentModel
from comp_model.core.data import BlockData, TrialDecision
from comp_model.core.events import EpisodeTrace
from comp_model.core.requirements import ComponentRequirements
from comp_model.plugins import PluginRegistry

from .compatibility import CompatibilityReport, assert_trace_compatible, check_trace_compatibility
from .likelihood import LikelihoodProgram
from .mle import ScipyMinimizeDiagnostics
from .transforms import ParameterTransform, identity_transform


@runtime_checkable
class PriorProgram(Protocol):
    """Protocol for parameter prior evaluators."""

    def log_prior(self, params: Mapping[str, float]) -> float:
        """Return total log-prior density for one parameter mapping."""


@dataclass(frozen=True, slots=True)
class PosteriorCandidate:
    """One posterior candidate evaluated by MAP optimization.

    Parameters
    ----------
    params : dict[str, float]
        Evaluated parameter set.
    log_likelihood : float
        Log-likelihood term for ``params``.
    log_prior : float
        Log-prior term for ``params``.
    log_posterior : float
        Total log-posterior value ``log_likelihood + log_prior``.
    """

    params: dict[str, float]
    log_likelihood: float
    log_prior: float
    log_posterior: float


@dataclass(frozen=True, slots=True)
class BayesFitResult:
    """MAP-based Bayesian fit output.

    Parameters
    ----------
    map_candidate : PosteriorCandidate
        Candidate with the maximum posterior value.
    candidates : tuple[PosteriorCandidate, ...]
        Candidate history captured during optimization.
    compatibility : CompatibilityReport | None
        Compatibility report when requirements were checked.
    scipy_diagnostics : ScipyMinimizeDiagnostics | None
        Diagnostics from SciPy minimization.
    """

    map_candidate: PosteriorCandidate
    candidates: tuple[PosteriorCandidate, ...]
    compatibility: CompatibilityReport | None = None
    scipy_diagnostics: ScipyMinimizeDiagnostics | None = None

    @property
    def map_params(self) -> dict[str, float]:
        """Return MAP parameter mapping."""

        return dict(self.map_candidate.params)


MapEstimatorType = Literal["scipy_map", "transformed_scipy_map"]


@dataclass(frozen=True, slots=True)
class MapFitSpec:
    """Estimator specification for MAP fitting.

    Parameters
    ----------
    estimator_type : {"scipy_map", "transformed_scipy_map"}
        Estimator backend.
    initial_params : dict[str, float]
        Initial constrained parameter values.
    bounds : dict[str, tuple[float | None, float | None]] | None, optional
        Constrained-space bounds for ``scipy_map``.
    bounds_z : dict[str, tuple[float | None, float | None]] | None, optional
        Unconstrained-space bounds for ``transformed_scipy_map``.
    transforms : dict[str, ParameterTransform] | None, optional
        Per-parameter transforms used by ``transformed_scipy_map``.
    method : str, optional
        SciPy optimizer method.
    tol : float | None, optional
        SciPy optimizer tolerance.
    """

    estimator_type: MapEstimatorType
    initial_params: dict[str, float]
    bounds: dict[str, tuple[float | None, float | None]] | None = None
    bounds_z: dict[str, tuple[float | None, float | None]] | None = None
    transforms: dict[str, ParameterTransform] | None = None
    method: str = "L-BFGS-B"
    tol: float | None = None


@dataclass(frozen=True, slots=True)
class IndependentPriorProgram:
    """Independent per-parameter prior program.

    Parameters
    ----------
    log_pdf_by_param : Mapping[str, Callable[[float], float]]
        Mapping from parameter name to scalar log-density callable.
    require_all : bool, optional
        If ``True``, every parameter in evaluated ``params`` must have a prior.

    Notes
    -----
    This class assumes independent priors:
    ``log p(theta) = sum_i log p_i(theta_i)``.
    """

    log_pdf_by_param: Mapping[str, Callable[[float], float]]
    require_all: bool = True

    def log_prior(self, params: Mapping[str, float]) -> float:
        """Return summed independent log-prior.

        Parameters
        ----------
        params : Mapping[str, float]
            Parameter mapping.

        Returns
        -------
        float
            Total log-prior. Non-finite component priors return ``-inf``.

        Raises
        ------
        ValueError
            If ``require_all`` is ``True`` and any parameter lacks a prior.
        """

        if self.require_all:
            missing = sorted(set(params) - set(self.log_pdf_by_param))
            if missing:
                raise ValueError(f"missing priors for parameters: {missing}")

        total = 0.0
        for name, value in params.items():
            log_pdf = self.log_pdf_by_param.get(name)
            if log_pdf is None:
                continue

            logp = float(log_pdf(float(value)))
            if not np.isfinite(logp):
                return float(-np.inf)
            total += logp

        return float(total)


def build_map_fit_function(
    *,
    model_factory: Callable[[dict[str, float]], AgentModel],
    prior_program: PriorProgram,
    fit_spec: MapFitSpec,
    requirements: ComponentRequirements | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> Callable[[EpisodeTrace], BayesFitResult]:
    """Build a reusable trace->MAP-fit function.

    Parameters
    ----------
    model_factory : Callable[[dict[str, float]], AgentModel]
        Factory creating a fresh model instance from parameters.
    prior_program : PriorProgram
        Prior evaluator.
    fit_spec : MapFitSpec
        MAP estimator specification.
    requirements : ComponentRequirements | None, optional
        Optional compatibility requirements checked before fitting.
    likelihood_program : LikelihoodProgram | None, optional
        Likelihood evaluator. Defaults to :class:`ActionReplayLikelihood`.

    Returns
    -------
    Callable[[EpisodeTrace], BayesFitResult]
        Function that fits one canonical trace.

    Raises
    ------
    ValueError
        If ``fit_spec`` is invalid.
    """

    raise RuntimeError(
        "SciPy Bayesian MAP estimators have been removed. "
        "Use Stan estimators via estimator.type='within_subject_hierarchical_stan_map' "
        "or estimator.type='within_subject_hierarchical_stan_nuts'."
    )


def fit_map_model(
    data: EpisodeTrace | BlockData | tuple[TrialDecision, ...] | list[TrialDecision],
    *,
    model_factory: Callable[[dict[str, float]], AgentModel],
    prior_program: PriorProgram,
    fit_spec: MapFitSpec,
    requirements: ComponentRequirements | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> BayesFitResult:
    """Fit one model with MAP using supported dataset containers.

    Parameters
    ----------
    data : EpisodeTrace | BlockData | tuple[TrialDecision, ...] | list[TrialDecision]
        Dataset container.
    model_factory : Callable[[dict[str, float]], AgentModel]
        Factory creating model instances from parameter mappings.
    prior_program : PriorProgram
        Prior evaluator.
    fit_spec : MapFitSpec
        MAP estimator specification.
    requirements : ComponentRequirements | None, optional
        Optional compatibility requirements checked before fitting.
    likelihood_program : LikelihoodProgram | None, optional
        Likelihood evaluator. Defaults to :class:`ActionReplayLikelihood`.

    Returns
    -------
    BayesFitResult
        MAP fit output.
    """

    raise RuntimeError(
        "fit_map_model is no longer supported. "
        "Use Stan hierarchical Bayesian APIs instead."
    )


def fit_map_model_from_registry(
    data: EpisodeTrace | BlockData | tuple[TrialDecision, ...] | list[TrialDecision],
    *,
    model_component_id: str,
    prior_program: PriorProgram,
    fit_spec: MapFitSpec,
    model_kwargs: Mapping[str, Any] | None = None,
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> BayesFitResult:
    """Fit one registered model component with MAP.

    Parameters
    ----------
    data : EpisodeTrace | BlockData | tuple[TrialDecision, ...] | list[TrialDecision]
        Dataset container.
    model_component_id : str
        Model component ID from the plugin registry.
    prior_program : PriorProgram
        Prior evaluator.
    fit_spec : MapFitSpec
        MAP estimator specification.
    model_kwargs : Mapping[str, Any] | None, optional
        Fixed model constructor keyword arguments.
    registry : PluginRegistry | None, optional
        Optional registry instance.
    likelihood_program : LikelihoodProgram | None, optional
        Likelihood evaluator. Defaults to :class:`ActionReplayLikelihood`.

    Returns
    -------
    BayesFitResult
        MAP fit output.
    """

    raise RuntimeError(
        "fit_map_model_from_registry is no longer supported. "
        "Use Stan hierarchical Bayesian APIs instead."
    )


def _merge_kwargs(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge fixed keyword arguments with free-parameter overrides."""

    merged = dict(base)
    merged.update(dict(override))
    return merged


def normal_log_prior(*, mean: float, std: float) -> Callable[[float], float]:
    """Build a Normal prior log-density function.

    Parameters
    ----------
    mean : float
        Normal distribution mean.
    std : float
        Positive standard deviation.

    Returns
    -------
    Callable[[float], float]
        Log-density function.
    """

    sigma = float(std)
    if sigma <= 0.0:
        raise ValueError("std must be > 0")

    mu = float(mean)
    log_norm = -0.5 * log(2.0 * pi * sigma * sigma)

    def log_pdf(value: float) -> float:
        centered = (float(value) - mu) / sigma
        return float(log_norm - 0.5 * centered * centered)

    return log_pdf


def uniform_log_prior(
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> Callable[[float], float]:
    """Build a uniform prior log-density function.

    Parameters
    ----------
    lower : float | None, optional
        Lower support bound.
    upper : float | None, optional
        Upper support bound.

    Returns
    -------
    Callable[[float], float]
        Log-density function returning ``-inf`` outside support.
    """

    lo = float(lower) if lower is not None else None
    hi = float(upper) if upper is not None else None
    if lo is not None and hi is not None and lo >= hi:
        raise ValueError("uniform prior requires lower < upper when both are provided")

    log_density = 0.0
    if lo is not None and hi is not None:
        log_density = -log(hi - lo)

    def log_pdf(value: float) -> float:
        v = float(value)
        if lo is not None and v < lo:
            return float(-np.inf)
        if hi is not None and v > hi:
            return float(-np.inf)
        return float(log_density)

    return log_pdf


def beta_log_prior(*, alpha: float, beta: float) -> Callable[[float], float]:
    """Build a Beta prior log-density function on ``(0, 1)``.

    Parameters
    ----------
    alpha : float
        Positive alpha shape parameter.
    beta : float
        Positive beta shape parameter.

    Returns
    -------
    Callable[[float], float]
        Log-density function.
    """

    a = float(alpha)
    b = float(beta)
    if a <= 0.0 or b <= 0.0:
        raise ValueError("alpha and beta must be > 0")

    log_norm = lgamma(a + b) - lgamma(a) - lgamma(b)

    def log_pdf(value: float) -> float:
        x = float(value)
        if x <= 0.0 or x >= 1.0:
            return float(-np.inf)
        return float(log_norm + (a - 1.0) * log(x) + (b - 1.0) * log1p(-x))

    return log_pdf


def log_normal_log_prior(*, mean_log: float, std_log: float) -> Callable[[float], float]:
    """Build a log-normal prior log-density function.

    Parameters
    ----------
    mean_log : float
        Mean of the underlying normal in log space.
    std_log : float
        Positive standard deviation in log space.

    Returns
    -------
    Callable[[float], float]
        Log-density function on positive reals.
    """

    sigma = float(std_log)
    if sigma <= 0.0:
        raise ValueError("std_log must be > 0")

    mu = float(mean_log)
    log_norm = -0.5 * log(2.0 * pi * sigma * sigma)

    def log_pdf(value: float) -> float:
        x = float(value)
        if x <= 0.0:
            return float(-np.inf)
        z = (log(x) - mu) / sigma
        return float(log_norm - log(x) - 0.5 * z * z)

    return log_pdf


class ScipyMapBayesEstimator:
    """MAP estimator using ``scipy.optimize.minimize``.

    Parameters
    ----------
    likelihood_program : LikelihoodProgram
        Replay-based likelihood evaluator.
    model_factory : Callable[[dict[str, float]], AgentModel]
        Factory creating a fresh model from parameter mappings.
    prior_program : PriorProgram
        Prior evaluator returning total log-prior density.
    requirements : ComponentRequirements | None, optional
        Optional compatibility requirements checked before fitting.
    method : str, optional
        SciPy minimization method.
    tol : float | None, optional
        SciPy tolerance.
    options : Mapping[str, Any] | None, optional
        Additional optimizer options.
    """

    def __init__(
        self,
        *,
        likelihood_program: LikelihoodProgram,
        model_factory: Callable[[dict[str, float]], AgentModel],
        prior_program: PriorProgram,
        requirements: ComponentRequirements | None = None,
        method: str = "L-BFGS-B",
        tol: float | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        self._likelihood_program = likelihood_program
        self._model_factory = model_factory
        self._prior_program = prior_program
        self._requirements = requirements
        self._method = str(method)
        self._tol = tol
        self._options = dict(options) if options is not None else None
        raise RuntimeError(
            "ScipyMapBayesEstimator has been removed. "
            "Use Stan Bayesian estimators instead."
        )

    def fit(
        self,
        trace: EpisodeTrace,
        *,
        initial_params: Mapping[str, float],
        bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
    ) -> BayesFitResult:
        """Run MAP optimization for one trace.

        Parameters
        ----------
        trace : EpisodeTrace
            Observed trace for inference.
        initial_params : Mapping[str, float]
            Initial parameter values.
        bounds : Mapping[str, tuple[float | None, float | None]] | None, optional
            Optional per-parameter box bounds.

        Returns
        -------
        BayesFitResult
            MAP candidate, evaluation history, and diagnostics.
        """

        compatibility: CompatibilityReport | None = None
        if self._requirements is not None:
            compatibility = check_trace_compatibility(trace, self._requirements)
            assert_trace_compatible(trace, self._requirements)

        names = tuple(sorted(initial_params))
        if not names:
            raise ValueError("initial_params must include at least one parameter")

        x0 = np.asarray([float(initial_params[name]) for name in names], dtype=float)
        scipy_bounds = _normalize_bounds(names, bounds)
        minimize = _load_scipy_minimize()

        candidates: list[PosteriorCandidate] = []

        def objective(x: np.ndarray) -> float:
            params = _vector_to_params(names, x)
            candidate = self._evaluate_candidate(trace=trace, params=params)
            candidates.append(candidate)

            if not np.isfinite(candidate.log_posterior):
                return 1e15
            return float(-candidate.log_posterior)

        result = minimize(
            objective,
            x0,
            method=self._method,
            bounds=scipy_bounds,
            tol=self._tol,
            options=self._options,
        )

        final_params = _vector_to_params(names, np.asarray(result.x, dtype=float))
        final_candidate = self._evaluate_candidate(trace=trace, params=final_params)
        candidates.append(final_candidate)

        map_candidate = max(candidates, key=lambda item: item.log_posterior)
        diagnostics = ScipyMinimizeDiagnostics(
            method=self._method,
            success=bool(result.success),
            status=int(result.status),
            message=str(result.message),
            n_iterations=int(getattr(result, "nit", -1)),
            n_function_evaluations=int(getattr(result, "nfev", len(candidates))),
        )
        return BayesFitResult(
            map_candidate=map_candidate,
            candidates=tuple(candidates),
            compatibility=compatibility,
            scipy_diagnostics=diagnostics,
        )

    def _evaluate_candidate(self, *, trace: EpisodeTrace, params: dict[str, float]) -> PosteriorCandidate:
        """Evaluate log-likelihood, log-prior, and log-posterior for ``params``."""

        model = self._model_factory(params)
        replay_result = self._likelihood_program.evaluate(trace, model)
        log_likelihood = float(replay_result.total_log_likelihood)
        log_prior = float(self._prior_program.log_prior(params))
        log_posterior = float(log_likelihood + log_prior)
        return PosteriorCandidate(
            params=dict(params),
            log_likelihood=log_likelihood,
            log_prior=log_prior,
            log_posterior=log_posterior,
        )


class TransformedScipyMapBayesEstimator:
    """MAP estimator with unconstrained optimization and parameter transforms.

    Parameters
    ----------
    likelihood_program : LikelihoodProgram
        Replay-based likelihood evaluator.
    model_factory : Callable[[dict[str, float]], AgentModel]
        Factory creating a fresh model from constrained parameter mappings.
    prior_program : PriorProgram
        Prior evaluator returning total log-prior density.
    transforms : Mapping[str, ParameterTransform] | None, optional
        Per-parameter transforms. Unspecified parameters use identity transform.
    requirements : ComponentRequirements | None, optional
        Optional compatibility requirements checked before fitting.
    method : str, optional
        SciPy minimization method.
    tol : float | None, optional
        SciPy tolerance.
    options : Mapping[str, Any] | None, optional
        Additional optimizer options.
    """

    def __init__(
        self,
        *,
        likelihood_program: LikelihoodProgram,
        model_factory: Callable[[dict[str, float]], AgentModel],
        prior_program: PriorProgram,
        transforms: Mapping[str, ParameterTransform] | None = None,
        requirements: ComponentRequirements | None = None,
        method: str = "L-BFGS-B",
        tol: float | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        self._likelihood_program = likelihood_program
        self._model_factory = model_factory
        self._prior_program = prior_program
        self._transforms = dict(transforms) if transforms is not None else {}
        self._requirements = requirements
        self._method = str(method)
        self._tol = tol
        self._options = dict(options) if options is not None else None
        raise RuntimeError(
            "TransformedScipyMapBayesEstimator has been removed. "
            "Use Stan Bayesian estimators instead."
        )

    def fit(
        self,
        trace: EpisodeTrace,
        *,
        initial_params: Mapping[str, float],
        bounds_z: Mapping[str, tuple[float | None, float | None]] | None = None,
    ) -> BayesFitResult:
        """Run transformed-space MAP optimization for one trace.

        Parameters
        ----------
        trace : EpisodeTrace
            Observed trace for inference.
        initial_params : Mapping[str, float]
            Initial constrained parameter values.
        bounds_z : Mapping[str, tuple[float | None, float | None]] | None, optional
            Optional per-parameter bounds in unconstrained space.

        Returns
        -------
        BayesFitResult
            MAP candidate, evaluation history, and diagnostics.
        """

        compatibility: CompatibilityReport | None = None
        if self._requirements is not None:
            compatibility = check_trace_compatibility(trace, self._requirements)
            assert_trace_compatible(trace, self._requirements)

        names = tuple(sorted(initial_params))
        if not names:
            raise ValueError("initial_params must include at least one parameter")

        transforms = {
            name: self._transforms.get(name, identity_transform())
            for name in names
        }

        z0 = np.asarray(
            [
                transforms[name].inverse(float(initial_params[name]))
                for name in names
            ],
            dtype=float,
        )
        scipy_bounds = _normalize_bounds(names, bounds_z)
        minimize = _load_scipy_minimize()

        candidates: list[PosteriorCandidate] = []

        def objective(z: np.ndarray) -> float:
            params = {
                name: float(transforms[name].forward(float(value)))
                for name, value in zip(names, z, strict=True)
            }
            candidate = self._evaluate_candidate(trace=trace, params=params)
            candidates.append(candidate)

            if not np.isfinite(candidate.log_posterior):
                return 1e15
            return float(-candidate.log_posterior)

        result = minimize(
            objective,
            z0,
            method=self._method,
            bounds=scipy_bounds,
            tol=self._tol,
            options=self._options,
        )

        final_z = np.asarray(result.x, dtype=float)
        final_params = {
            name: float(transforms[name].forward(float(value)))
            for name, value in zip(names, final_z, strict=True)
        }
        final_candidate = self._evaluate_candidate(trace=trace, params=final_params)
        candidates.append(final_candidate)

        map_candidate = max(candidates, key=lambda item: item.log_posterior)
        diagnostics = ScipyMinimizeDiagnostics(
            method=self._method,
            success=bool(result.success),
            status=int(result.status),
            message=str(result.message),
            n_iterations=int(getattr(result, "nit", -1)),
            n_function_evaluations=int(getattr(result, "nfev", len(candidates))),
        )
        return BayesFitResult(
            map_candidate=map_candidate,
            candidates=tuple(candidates),
            compatibility=compatibility,
            scipy_diagnostics=diagnostics,
        )

    def _evaluate_candidate(self, *, trace: EpisodeTrace, params: dict[str, float]) -> PosteriorCandidate:
        """Evaluate log-likelihood, log-prior, and log-posterior for ``params``."""

        model = self._model_factory(params)
        replay_result = self._likelihood_program.evaluate(trace, model)
        log_likelihood = float(replay_result.total_log_likelihood)
        log_prior = float(self._prior_program.log_prior(params))
        log_posterior = float(log_likelihood + log_prior)
        return PosteriorCandidate(
            params=dict(params),
            log_likelihood=log_likelihood,
            log_prior=log_prior,
            log_posterior=log_posterior,
        )


def _normalize_bounds(
    names: tuple[str, ...],
    bounds: Mapping[str, tuple[float | None, float | None]] | None,
) -> tuple[tuple[float | None, float | None], ...]:
    """Normalize and validate per-parameter bounds."""

    if bounds is None:
        return tuple((None, None) for _ in names)

    unknown = sorted(set(bounds) - set(names))
    if unknown:
        raise ValueError(f"bounds include unknown parameters: {unknown}")

    out: list[tuple[float | None, float | None]] = []
    for name in names:
        lower, upper = bounds.get(name, (None, None))
        low = float(lower) if lower is not None else None
        high = float(upper) if upper is not None else None
        if low is not None and high is not None and low > high:
            raise ValueError(f"invalid bounds for {name!r}: lower ({low}) > upper ({high})")
        out.append((low, high))
    return tuple(out)


def _vector_to_params(names: tuple[str, ...], vector: np.ndarray) -> dict[str, float]:
    """Convert ordered parameter vector into a named mapping."""

    return {name: float(value) for name, value in zip(names, vector, strict=True)}


def _load_scipy_minimize() -> Callable[..., Any]:
    """Import and return ``scipy.optimize.minimize``."""

    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover - exercised only without scipy installed
        raise ImportError(
            "Bayesian MAP estimators require scipy. Install with `pip install scipy`."
        ) from exc
    return minimize


__all__ = [
    "BayesFitResult",
    "IndependentPriorProgram",
    "MapEstimatorType",
    "MapFitSpec",
    "PosteriorCandidate",
    "PriorProgram",
    "ScipyMapBayesEstimator",
    "TransformedScipyMapBayesEstimator",
    "beta_log_prior",
    "build_map_fit_function",
    "fit_map_model",
    "fit_map_model_from_registry",
    "log_normal_log_prior",
    "normal_log_prior",
    "uniform_log_prior",
]
