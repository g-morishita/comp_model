"""Reusable model-fitting helpers for user datasets and recovery pipelines."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from comp_model.core.contracts import AgentModel
from comp_model.core.data import (
    BlockData,
    TrialDecision,
    get_block_trace,
    trace_from_trial_decisions,
)
from comp_model.core.events import EpisodeTrace, validate_trace
from comp_model.core.requirements import ComponentRequirements
from comp_model.plugins import PluginRegistry, build_default_registry

from .likelihood import ActionReplayLikelihood, LikelihoodProgram
from .mle import (
    GridSearchMLEEstimator,
    MLEFitResult,
    ScipyMinimizeMLEEstimator,
    TransformedScipyMinimizeMLEEstimator,
)
from .transforms import ParameterTransform

FitInferenceType = Literal["mle", "bayesian"]
MLESolverType = Literal["grid_search", "scipy_minimize", "transformed_scipy_minimize"]


@dataclass(frozen=True, slots=True)
class FitSpec:
    """Estimator specification for model fitting.

    Parameters
    ----------
    inference : {"mle", "bayesian"}, optional
        High-level inference family. ``fit_model`` currently supports
        ``"mle"`` only; ``"bayesian"`` is routed through Stan posterior APIs.
    solver : {"grid_search", "scipy_minimize", "transformed_scipy_minimize"} | None, optional
        Concrete MLE solver/backend. If omitted, a solver is chosen
        automatically:
        - ``"grid_search"`` when ``parameter_grid`` is provided,
        - otherwise ``"scipy_minimize"``.
    parameter_grid : dict[str, list[float]] | None, optional
        Required for ``grid_search``.
    initial_params : dict[str, float] | None, optional
        Required for SciPy-based estimators.
    bounds : dict[str, tuple[float | None, float | None]] | None, optional
        Parameter-space bounds for ``scipy_minimize``.
    bounds_z : dict[str, tuple[float | None, float | None]] | None, optional
        Unconstrained-space bounds for transformed SciPy MLE.
    transforms : dict[str, ParameterTransform] | None, optional
        Per-parameter transforms for transformed SciPy MLE.
    method : str, optional
        SciPy optimizer method.
    tol : float | None, optional
        SciPy optimizer tolerance.
    n_starts : int, optional
        Number of optimizer starts for SciPy-based MLE.
        Multiple starts are evaluated and the best likelihood is retained.
    random_seed : int | None, optional
        Random seed used to generate additional randomized starts.
        Set to ``None`` to disable deterministic seeding.
    """

    inference: FitInferenceType = "mle"
    solver: MLESolverType | None = None
    parameter_grid: dict[str, list[float]] | None = None
    initial_params: dict[str, float] | None = None
    bounds: dict[str, tuple[float | None, float | None]] | None = None
    bounds_z: dict[str, tuple[float | None, float | None]] | None = None
    transforms: dict[str, ParameterTransform] | None = None
    method: str = "L-BFGS-B"
    tol: float | None = None
    n_starts: int = 5
    random_seed: int | None = 0


def coerce_episode_trace(data: EpisodeTrace | BlockData | Sequence[TrialDecision]) -> EpisodeTrace:
    """Coerce supported dataset containers into canonical ``EpisodeTrace``.

    Parameters
    ----------
    data : EpisodeTrace | BlockData | Sequence[TrialDecision]
        Input dataset.

    Returns
    -------
    EpisodeTrace
        Canonical trace used by likelihood/replay.

    Raises
    ------
    TypeError
        If data type is unsupported.
    ValueError
        If conversion fails (for example empty decision sequence).
    """

    if isinstance(data, EpisodeTrace):
        validate_trace(data)
        return data

    if isinstance(data, BlockData):
        return get_block_trace(data)

    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        decisions = tuple(data)
        if len(decisions) == 0:
            raise ValueError("decision sequence must not be empty")
        if not all(isinstance(item, TrialDecision) for item in decisions):
            raise TypeError("decision sequences must contain TrialDecision items")
        return trace_from_trial_decisions(decisions)

    raise TypeError(
        "data must be EpisodeTrace, BlockData, or a non-empty sequence of TrialDecision"
    )


def build_model_fit_function(
    *,
    model_factory: Callable[[dict[str, float]], AgentModel],
    fit_spec: FitSpec,
    requirements: ComponentRequirements | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> Callable[[EpisodeTrace], MLEFitResult]:
    """Build a reusable trace->fit callable from estimator specification.

    Parameters
    ----------
    model_factory : Callable[[dict[str, float]], AgentModel]
        Factory building a model instance from parameter mapping.
    fit_spec : FitSpec
        Estimator specification.
    requirements : ComponentRequirements | None, optional
        Optional compatibility requirements checked before fitting.
    likelihood_program : LikelihoodProgram | None, optional
        Likelihood program. Defaults to :class:`ActionReplayLikelihood`.

    Returns
    -------
    Callable[[EpisodeTrace], MLEFitResult]
        Function that fits a canonical episode trace.

    Raises
    ------
    ValueError
        If ``fit_spec`` is invalid for the selected estimator.
    """

    likelihood = likelihood_program if likelihood_program is not None else ActionReplayLikelihood()

    solver = _resolve_mle_solver(fit_spec)

    if solver == "grid_search":
        if fit_spec.parameter_grid is None:
            raise ValueError("fit_spec.parameter_grid is required for grid_search")

        grid_estimator = GridSearchMLEEstimator(
            likelihood_program=likelihood,
            model_factory=model_factory,
            requirements=requirements,
        )
        return lambda trace: grid_estimator.fit(trace=trace, parameter_grid=fit_spec.parameter_grid or {})

    if solver == "scipy_minimize":
        if fit_spec.initial_params is None:
            raise ValueError("fit_spec.initial_params is required for scipy_minimize")

        scipy_estimator = ScipyMinimizeMLEEstimator(
            likelihood_program=likelihood,
            model_factory=model_factory,
            requirements=requirements,
            method=fit_spec.method,
            tol=fit_spec.tol,
        )
        return lambda trace: scipy_estimator.fit(
            trace=trace,
            initial_params=fit_spec.initial_params or {},
            bounds=fit_spec.bounds,
            n_starts=fit_spec.n_starts,
            random_seed=fit_spec.random_seed,
        )

    if solver == "transformed_scipy_minimize":
        if fit_spec.initial_params is None:
            raise ValueError("fit_spec.initial_params is required for transformed_scipy_minimize")

        transformed_estimator = TransformedScipyMinimizeMLEEstimator(
            likelihood_program=likelihood,
            model_factory=model_factory,
            transforms=fit_spec.transforms,
            requirements=requirements,
            method=fit_spec.method,
            tol=fit_spec.tol,
        )
        return lambda trace: transformed_estimator.fit(
            trace=trace,
            initial_params=fit_spec.initial_params or {},
            bounds_z=fit_spec.bounds_z,
            n_starts=fit_spec.n_starts,
            random_seed=fit_spec.random_seed,
        )

    raise ValueError(
        "fit_spec.solver must be one of "
        "{'grid_search', 'scipy_minimize', 'transformed_scipy_minimize'}"
    )


def _resolve_mle_solver(fit_spec: FitSpec) -> MLESolverType:
    """Resolve high-level fit spec into one concrete MLE solver.

    Parameters
    ----------
    fit_spec : FitSpec
        Fit specification containing inference family and optional solver hints.

    Returns
    -------
    MLESolverType
        Concrete solver name used by MLE fitting.

    Raises
    ------
    ValueError
        If inference type is unsupported in ``fit_model`` or solver hints
        are contradictory.
    """

    if fit_spec.inference == "bayesian":
        raise ValueError(
            "fit_model currently supports only inference='mle'. "
            "Use Stan Bayesian APIs (for example, sample_subject_hierarchical_posterior_stan)."
        )
    if fit_spec.inference != "mle":
        raise ValueError("fit_spec.inference must be either 'mle' or 'bayesian'")

    solver = fit_spec.solver
    if solver is None:
        if fit_spec.parameter_grid is not None:
            solver = "grid_search"
        else:
            solver = "scipy_minimize"

    return solver


def fit_model(
    data: EpisodeTrace | BlockData | Sequence[TrialDecision],
    *,
    model_factory: Callable[[dict[str, float]], AgentModel],
    fit_spec: FitSpec,
    requirements: ComponentRequirements | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> MLEFitResult:
    """Fit a model to supported dataset containers.

    Parameters
    ----------
    data : EpisodeTrace | BlockData | Sequence[TrialDecision]
        Input dataset to fit.
    model_factory : Callable[[dict[str, float]], AgentModel]
        Factory building model instances from parameter mappings.
    fit_spec : FitSpec
        Estimator specification.
    requirements : ComponentRequirements | None, optional
        Optional compatibility requirements checked before fitting.
    likelihood_program : LikelihoodProgram | None, optional
        Likelihood program. Defaults to :class:`ActionReplayLikelihood`.

    Returns
    -------
    MLEFitResult
        Fitting result.
    """

    trace = coerce_episode_trace(data)
    fit_function = build_model_fit_function(
        model_factory=model_factory,
        fit_spec=fit_spec,
        requirements=requirements,
        likelihood_program=likelihood_program,
    )
    return fit_function(trace)


def fit_model_from_registry(
    data: EpisodeTrace | BlockData | Sequence[TrialDecision],
    *,
    model_component_id: str,
    fit_spec: FitSpec,
    model_kwargs: Mapping[str, Any] | None = None,
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> MLEFitResult:
    """Fit one registered model component to a dataset.

    Parameters
    ----------
    data : EpisodeTrace | BlockData | Sequence[TrialDecision]
        Input dataset to fit.
    model_component_id : str
        Model component ID in the plugin registry.
    fit_spec : FitSpec
        Estimator specification.
    model_kwargs : Mapping[str, Any] | None, optional
        Fixed model kwargs applied for every candidate before parameter updates.
    registry : PluginRegistry | None, optional
        Optional plugin registry instance. Defaults to built-in registry.
    likelihood_program : LikelihoodProgram | None, optional
        Likelihood evaluator. Defaults to :class:`ActionReplayLikelihood`.

    Returns
    -------
    MLEFitResult
        Fitting result.
    """

    reg = registry if registry is not None else build_default_registry()
    manifest = reg.get("model", model_component_id)
    fixed_kwargs = dict(model_kwargs) if model_kwargs is not None else {}

    return fit_model(
        data,
        model_factory=lambda params: reg.create_model(
            model_component_id,
            **_merge_kwargs(fixed_kwargs, params),
        ),
        fit_spec=fit_spec,
        requirements=manifest.requirements,
        likelihood_program=likelihood_program,
    )


def _merge_kwargs(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge fixed kwargs and per-candidate parameter kwargs."""

    merged = dict(base)
    merged.update(override)
    return merged


__all__ = [
    "FitInferenceType",
    "FitSpec",
    "MLESolverType",
    "build_model_fit_function",
    "coerce_episode_trace",
    "fit_model",
    "fit_model_from_registry",
]
