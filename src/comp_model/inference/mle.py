"""Maximum-likelihood estimation interfaces and baseline implementations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable

import numpy as np

from comp_model.core.contracts import AgentModel
from comp_model.core.events import EpisodeTrace
from comp_model.core.requirements import ComponentRequirements
from comp_model.inference.compatibility import CompatibilityReport, assert_trace_compatible, check_trace_compatibility
from comp_model.inference.likelihood import LikelihoodProgram


@dataclass(frozen=True, slots=True)
class MLECandidate:
    """One parameter candidate evaluated by an MLE estimator.

    Parameters
    ----------
    params : dict[str, float]
        Evaluated parameter set.
    log_likelihood : float
        Total action log-likelihood for ``params``.
    """

    params: dict[str, float]
    log_likelihood: float


@dataclass(frozen=True, slots=True)
class ScipyMinimizeDiagnostics:
    """Diagnostics produced by ``scipy.optimize.minimize``.

    Parameters
    ----------
    method : str
        Scipy optimization method used.
    success : bool
        Whether SciPy reported successful termination.
    status : int
        SciPy status code.
    message : str
        SciPy termination message.
    n_iterations : int
        Number of optimizer iterations.
    n_function_evaluations : int
        Number of objective evaluations.
    """

    method: str
    success: bool
    status: int
    message: str
    n_iterations: int
    n_function_evaluations: int


@dataclass(frozen=True, slots=True)
class MLEFitResult:
    """MLE fit output.

    Parameters
    ----------
    best : MLECandidate
        Candidate with the maximum log-likelihood.
    candidates : tuple[MLECandidate, ...]
        All evaluated candidates.
    compatibility : CompatibilityReport | None
        Compatibility report when requirements were checked.
    scipy_diagnostics : ScipyMinimizeDiagnostics | None
        SciPy minimization diagnostics when optimizer-backed MLE is used.
    """

    best: MLECandidate
    candidates: tuple[MLECandidate, ...]
    compatibility: CompatibilityReport | None = None
    scipy_diagnostics: ScipyMinimizeDiagnostics | None = None


class GridSearchMLEEstimator:
    """Deterministic grid-search MLE estimator.

    Parameters
    ----------
    likelihood_program : LikelihoodProgram
        Likelihood evaluator used for each candidate model.
    model_factory : Callable[[dict[str, float]], AgentModel]
        Factory creating a fresh model instance for one candidate parameter set.
    requirements : ComponentRequirements | None, optional
        Optional compatibility requirements checked before fitting.

    Notes
    -----
    This implementation is intentionally simple and serves as a baseline API
    for future optimizer-backed estimators.
    """

    def __init__(
        self,
        likelihood_program: LikelihoodProgram,
        model_factory: Callable[[dict[str, float]], AgentModel],
        requirements: ComponentRequirements | None = None,
    ) -> None:
        self._likelihood_program = likelihood_program
        self._model_factory = model_factory
        self._requirements = requirements

    def fit(self, trace: EpisodeTrace, parameter_grid: dict[str, list[float]]) -> MLEFitResult:
        """Fit model parameters via exhaustive grid search.

        Parameters
        ----------
        trace : EpisodeTrace
            Observed event trace used for likelihood evaluation.
        parameter_grid : dict[str, list[float]]
            Parameter grid values. Every combination is evaluated.

        Returns
        -------
        MLEFitResult
            Full candidate list and best-fit parameters.

        Raises
        ------
        ValueError
            If compatibility fails or the parameter grid is empty.
        """

        compatibility: CompatibilityReport | None = None
        if self._requirements is not None:
            compatibility = check_trace_compatibility(trace, self._requirements)
            assert_trace_compatible(trace, self._requirements)

        candidates: list[MLECandidate] = []
        for params in _iter_parameter_grid(parameter_grid):
            model = self._model_factory(params)
            replay_result = self._likelihood_program.evaluate(trace, model)
            candidates.append(
                MLECandidate(
                    params=dict(params),
                    log_likelihood=float(replay_result.total_log_likelihood),
                )
            )

        if not candidates:
            raise ValueError("parameter_grid must include at least one candidate")

        best = max(candidates, key=lambda item: item.log_likelihood)
        return MLEFitResult(best=best, candidates=tuple(candidates), compatibility=compatibility)


class ScipyMinimizeMLEEstimator:
    """Optimizer-backed MLE estimator using ``scipy.optimize.minimize``.

    Parameters
    ----------
    likelihood_program : LikelihoodProgram
        Likelihood evaluator used for each candidate model.
    model_factory : Callable[[dict[str, float]], AgentModel]
        Factory creating a fresh model instance for one candidate parameter set.
    requirements : ComponentRequirements | None, optional
        Optional compatibility requirements checked before fitting.
    method : str, optional
        Scipy minimization method (default ``"L-BFGS-B"``).
    tol : float | None, optional
        Termination tolerance passed to SciPy.
    options : Mapping[str, Any] | None, optional
        Additional optimizer options passed to SciPy.

    Notes
    -----
    The estimator maximizes log-likelihood by minimizing negative
    log-likelihood. Bounds are provided as box constraints per parameter.
    """

    def __init__(
        self,
        likelihood_program: LikelihoodProgram,
        model_factory: Callable[[dict[str, float]], AgentModel],
        requirements: ComponentRequirements | None = None,
        *,
        method: str = "L-BFGS-B",
        tol: float | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        self._likelihood_program = likelihood_program
        self._model_factory = model_factory
        self._requirements = requirements
        self._method = str(method)
        self._tol = tol
        self._options = dict(options) if options is not None else None

    def fit(
        self,
        trace: EpisodeTrace,
        initial_params: Mapping[str, float],
        *,
        bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
    ) -> MLEFitResult:
        """Fit model parameters using SciPy minimization.

        Parameters
        ----------
        trace : EpisodeTrace
            Observed event trace used for likelihood evaluation.
        initial_params : Mapping[str, float]
            Initial parameter values.
        bounds : Mapping[str, tuple[float | None, float | None]] | None, optional
            Box bounds by parameter name. Missing bounds default to ``(None, None)``.

        Returns
        -------
        MLEFitResult
            Fit result with best candidate, evaluation history, and diagnostics.

        Raises
        ------
        ValueError
            If compatibility fails, parameters are empty, or bounds are invalid.
        ImportError
            If SciPy is not installed.
        """

        compatibility: CompatibilityReport | None = None
        if self._requirements is not None:
            compatibility = check_trace_compatibility(trace, self._requirements)
            assert_trace_compatible(trace, self._requirements)

        minimize = _load_scipy_minimize()

        names = tuple(sorted(initial_params))
        if not names:
            raise ValueError("initial_params must include at least one parameter")

        x0 = np.asarray([float(initial_params[name]) for name in names], dtype=float)
        scipy_bounds = _normalize_scipy_bounds(names, bounds)

        candidates: list[MLECandidate] = []

        def objective(x: np.ndarray) -> float:
            params = _vector_to_params(names, x)
            model = self._model_factory(params)
            replay_result = self._likelihood_program.evaluate(trace, model)
            log_likelihood = float(replay_result.total_log_likelihood)
            candidates.append(MLECandidate(params=params, log_likelihood=log_likelihood))

            if not np.isfinite(log_likelihood):
                return 1e15
            return -log_likelihood

        result = minimize(
            objective,
            x0,
            method=self._method,
            bounds=scipy_bounds,
            tol=self._tol,
            options=self._options,
        )

        final_params = _vector_to_params(names, np.asarray(result.x, dtype=float))
        final_model = self._model_factory(final_params)
        final_replay = self._likelihood_program.evaluate(trace, final_model)
        final_candidate = MLECandidate(
            params=final_params,
            log_likelihood=float(final_replay.total_log_likelihood),
        )
        candidates.append(final_candidate)

        best = max(candidates, key=lambda item: item.log_likelihood)
        diagnostics = ScipyMinimizeDiagnostics(
            method=self._method,
            success=bool(result.success),
            status=int(result.status),
            message=str(result.message),
            n_iterations=int(getattr(result, "nit", -1)),
            n_function_evaluations=int(getattr(result, "nfev", len(candidates))),
        )
        return MLEFitResult(
            best=best,
            candidates=tuple(candidates),
            compatibility=compatibility,
            scipy_diagnostics=diagnostics,
        )


def _load_scipy_minimize() -> Callable[..., Any]:
    """Import and return ``scipy.optimize.minimize``.

    Returns
    -------
    Callable[..., Any]
        SciPy minimize function.

    Raises
    ------
    ImportError
        If SciPy is not installed.
    """

    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover - exercised only without scipy installed
        raise ImportError(
            "ScipyMinimizeMLEEstimator requires scipy. Install with `pip install scipy`."
        ) from exc
    return minimize


def _normalize_scipy_bounds(
    names: tuple[str, ...],
    bounds: Mapping[str, tuple[float | None, float | None]] | None,
) -> tuple[tuple[float | None, float | None], ...]:
    """Normalize and validate per-parameter SciPy bounds.

    Parameters
    ----------
    names : tuple[str, ...]
        Ordered parameter names.
    bounds : Mapping[str, tuple[float | None, float | None]] | None
        Raw bound mapping.

    Returns
    -------
    tuple[tuple[float | None, float | None], ...]
        Bounds in ``names`` order.

    Raises
    ------
    ValueError
        If unknown parameters are present or lower bound exceeds upper bound.
    """

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
    """Convert ordered parameter vector to named parameter mapping."""

    return {name: float(value) for name, value in zip(names, vector, strict=True)}


def _iter_parameter_grid(parameter_grid: dict[str, list[float]]) -> tuple[dict[str, float], ...]:
    """Generate deterministic parameter combinations from a grid.

    Parameters
    ----------
    parameter_grid : dict[str, list[float]]
        Mapping from parameter names to candidate values.

    Returns
    -------
    tuple[dict[str, float], ...]
        Candidate parameter dictionaries in deterministic key order.
    """

    if not parameter_grid:
        return tuple()

    names = tuple(sorted(parameter_grid))
    values = []
    for name in names:
        grid_values = tuple(float(v) for v in parameter_grid[name])
        if not grid_values:
            raise ValueError(f"parameter {name!r} has no candidate values")
        values.append(grid_values)

    combinations = []
    for candidate_values in product(*values):
        combinations.append({name: value for name, value in zip(names, candidate_values, strict=True)})
    return tuple(combinations)
