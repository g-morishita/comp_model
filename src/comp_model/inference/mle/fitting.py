"""Reusable MLE fitting helpers for traces, blocks, and recovery pipelines."""

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

from ..likelihood import ActionReplayLikelihood, LikelihoodProgram
from .estimators import (
    GridSearchMLEEstimator,
    MLEFitResult,
    ScipyMinimizeMLEEstimator,
    TransformedScipyMinimizeMLEEstimator,
)
from ..transforms import ParameterTransform

MLESolverType = Literal["grid_search", "scipy_minimize", "transformed_scipy_minimize"]


@dataclass(frozen=True, slots=True)
class MLEFitSpec:
    """Estimator specification for maximum-likelihood fitting.

    Parameters
    ----------
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
    """Coerce supported trace-like containers into canonical ``EpisodeTrace``.

    Parameters
    ----------
    data : EpisodeTrace | BlockData | Sequence[TrialDecision]
        Input trace-like container.

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


def coerce_episode_traces(
    data_items: Sequence[EpisodeTrace | BlockData | Sequence[TrialDecision]],
) -> tuple[EpisodeTrace, ...]:
    """Coerce one or more trace-like containers into canonical traces."""

    if not isinstance(data_items, Sequence) or isinstance(data_items, (str, bytes, bytearray)):
        raise TypeError(
            "data_items must be a non-empty sequence of EpisodeTrace, BlockData, "
            "or non-empty sequences of TrialDecision"
        )

    items = tuple(data_items)
    if not items:
        raise ValueError("data_items must include at least one trace-like container")

    return tuple(coerce_episode_trace(item) for item in items)


def _build_joint_trace_fit_function(
    *,
    model_factory: Callable[[dict[str, float]], AgentModel],
    fit_spec: MLEFitSpec,
    requirements: ComponentRequirements | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> Callable[[tuple[EpisodeTrace, ...]], MLEFitResult]:
    """Build a reusable traces->fit callable for shared-parameter fitting."""

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
        return lambda traces: grid_estimator.fit_traces(
            traces=traces,
            parameter_grid=fit_spec.parameter_grid or {},
        )

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
        return lambda traces: scipy_estimator.fit_traces(
            traces=traces,
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
        return lambda traces: transformed_estimator.fit_traces(
            traces=traces,
            initial_params=fit_spec.initial_params or {},
            bounds_z=fit_spec.bounds_z,
            n_starts=fit_spec.n_starts,
            random_seed=fit_spec.random_seed,
        )

    raise ValueError(
        "fit_spec.solver must be one of "
        "{'grid_search', 'scipy_minimize', 'transformed_scipy_minimize'}"
    )


def _build_trace_fit_function(
    *,
    model_factory: Callable[[dict[str, float]], AgentModel],
    fit_spec: MLEFitSpec,
    requirements: ComponentRequirements | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> Callable[[EpisodeTrace], MLEFitResult]:
    """Build a reusable trace->fit callable from estimator specification.

    Parameters
    ----------
    model_factory : Callable[[dict[str, float]], AgentModel]
        Factory building a model instance from parameter mapping.
    fit_spec : MLEFitSpec
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

    joint_fit = _build_joint_trace_fit_function(
        model_factory=model_factory,
        fit_spec=fit_spec,
        requirements=requirements,
        likelihood_program=likelihood_program,
    )
    return lambda trace: joint_fit((trace,))


def _resolve_mle_solver(fit_spec: MLEFitSpec) -> MLESolverType:
    """Resolve one MLE fit spec into one concrete solver.

    Parameters
    ----------
    fit_spec : MLEFitSpec
        Fit specification containing optional solver hints.

    Returns
    -------
    MLESolverType
        Concrete solver name used by MLE fitting.

    Raises
    ------
    ValueError
        If solver hints are contradictory.
    """

    solver = fit_spec.solver
    if solver is None:
        if fit_spec.parameter_grid is not None:
            solver = "grid_search"
        else:
            solver = "scipy_minimize"

    return solver


def fit_trace(
    data: EpisodeTrace | BlockData | Sequence[TrialDecision],
    *,
    model_factory: Callable[[dict[str, float]], AgentModel],
    fit_spec: MLEFitSpec,
    requirements: ComponentRequirements | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> MLEFitResult:
    """Fit a model to supported trace-like containers.

    Parameters
    ----------
    data : EpisodeTrace | BlockData | Sequence[TrialDecision]
        Input trace-like container to fit.
    model_factory : Callable[[dict[str, float]], AgentModel]
        Factory building model instances from parameter mappings.
    fit_spec : MLEFitSpec
        MLE estimator specification.
    requirements : ComponentRequirements | None, optional
        Optional compatibility requirements checked before fitting.
    likelihood_program : LikelihoodProgram | None, optional
        Likelihood program. Defaults to :class:`ActionReplayLikelihood`.

    Returns
    -------
    MLEFitResult
        Fitting result.
    """

    return fit_joint_traces(
        (data,),
        model_factory=model_factory,
        fit_spec=fit_spec,
        requirements=requirements,
        likelihood_program=likelihood_program,
    )


def fit_joint_traces(
    data_items: Sequence[EpisodeTrace | BlockData | Sequence[TrialDecision]],
    *,
    model_factory: Callable[[dict[str, float]], AgentModel],
    fit_spec: MLEFitSpec,
    requirements: ComponentRequirements | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> MLEFitResult:
    """Fit one shared parameter set across multiple trace-like containers.

    Parameters
    ----------
    data_items : Sequence[EpisodeTrace | BlockData | Sequence[TrialDecision]]
        One or more trace-like containers fit jointly with shared parameters.
    model_factory : Callable[[dict[str, float]], AgentModel]
        Factory building model instances from parameter mappings.
    fit_spec : MLEFitSpec
        MLE estimator specification.
    requirements : ComponentRequirements | None, optional
        Optional compatibility requirements checked for every trace.
    likelihood_program : LikelihoodProgram | None, optional
        Likelihood program. Defaults to :class:`ActionReplayLikelihood`.

    Returns
    -------
    MLEFitResult
        Shared-parameter fitting result.
    """

    traces = coerce_episode_traces(data_items)
    fit_function = _build_joint_trace_fit_function(
        model_factory=model_factory,
        fit_spec=fit_spec,
        requirements=requirements,
        likelihood_program=likelihood_program,
    )
    return fit_function(traces)


def fit_trace_from_registry(
    data: EpisodeTrace | BlockData | Sequence[TrialDecision],
    *,
    model_component_id: str,
    fit_spec: MLEFitSpec,
    model_kwargs: Mapping[str, Any] | None = None,
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> MLEFitResult:
    """Fit one registered model component to a trace-like container.

    Parameters
    ----------
    data : EpisodeTrace | BlockData | Sequence[TrialDecision]
        Input trace-like container to fit.
    model_component_id : str
        Model component ID in the plugin registry.
    fit_spec : MLEFitSpec
        MLE estimator specification.
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

    return fit_trace(
        data,
        model_factory=lambda params: reg.create_model(
            model_component_id,
            **_merge_kwargs(fixed_kwargs, params),
        ),
        fit_spec=fit_spec,
        requirements=manifest.requirements,
        likelihood_program=likelihood_program,
    )


def fit_joint_traces_from_registry(
    data_items: Sequence[EpisodeTrace | BlockData | Sequence[TrialDecision]],
    *,
    model_component_id: str,
    fit_spec: MLEFitSpec,
    model_kwargs: Mapping[str, Any] | None = None,
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> MLEFitResult:
    """Fit one registered model jointly across multiple trace-like containers."""

    reg = registry if registry is not None else build_default_registry()
    manifest = reg.get("model", model_component_id)
    fixed_kwargs = dict(model_kwargs) if model_kwargs is not None else {}

    return fit_joint_traces(
        data_items,
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
    "MLEFitSpec",
    "MLESolverType",
    "coerce_episode_trace",
    "coerce_episode_traces",
    "fit_joint_traces",
    "fit_joint_traces_from_registry",
    "fit_trace",
    "fit_trace_from_registry",
]
