"""Config-driven model fitting helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from comp_model.core.config_validation import validate_allowed_keys
from comp_model.core.data import BlockData, StudyData, SubjectData, TrialDecision
from comp_model.core.events import EpisodeTrace
from comp_model.plugins import PluginRegistry, build_default_registry

from .block_strategy import coerce_block_fit_strategy
from .fitting import FitInferenceType, FitSpec, MLESolverType, fit_model_from_registry
from .likelihood_config import likelihood_program_from_config
from .mle import MLEFitResult
from .study_fitting import (
    BlockFitResult,
    BlockFitStrategy,
    StudyFitResult,
    SubjectFitResult,
    fit_block_data,
    fit_study_data,
    fit_subject_data,
)
from .transforms import (
    ParameterTransform,
    identity_transform,
    positive_log_transform,
    unit_interval_logit_transform,
)


@dataclass(frozen=True, slots=True)
class ModelComponentSpec:
    """Model component spec parsed from config.

    Parameters
    ----------
    component_id : str
        Model component ID in plugin registry.
    kwargs : dict[str, Any]
        Fixed model constructor kwargs.
    """

    component_id: str
    kwargs: dict[str, Any]


def fit_spec_from_config(estimator_cfg: Mapping[str, Any]) -> FitSpec:
    """Parse estimator config mapping into :class:`FitSpec`.

    Parameters
    ----------
    estimator_cfg : Mapping[str, Any]
        Estimator configuration mapping.

    Returns
    -------
    FitSpec
        Parsed fit specification.

    Raises
    ------
    ValueError
        If required fields are missing or invalid.
    """

    estimator = _require_mapping(estimator_cfg, field_name="estimator")
    inference_raw = _coerce_non_empty_str(estimator.get("type"), field_name="estimator.type")
    inference = _coerce_fit_inference(inference_raw, field_name="estimator.type")
    if inference == "bayesian":
        raise ValueError(
            "fit_spec_from_config currently supports only estimator.type='mle'. "
            "For Bayesian inference, use Stan hierarchical estimator types "
            "with fit_*_auto_from_config."
        )

    solver_raw = estimator.get("solver")
    if solver_raw is not None:
        solver_name = _coerce_non_empty_str(solver_raw, field_name="estimator.solver")
        solver = _coerce_mle_solver(solver_name, field_name="estimator.solver")
    else:
        if "parameter_grid" in estimator:
            solver = "grid_search"
        elif "bounds_z" in estimator or "transforms" in estimator:
            solver = "transformed_scipy_minimize"
        else:
            solver = "scipy_minimize"

    _validate_estimator_keys_for_solver(
        estimator,
        solver=solver,
        allow_solver_key=True,
    )

    return FitSpec(
        inference=inference,
        solver=solver,
        parameter_grid=(
            _coerce_float_list_mapping(estimator.get("parameter_grid"), field_name="estimator.parameter_grid")
            if solver == "grid_search"
            else None
        ),
        initial_params=(
            _coerce_float_mapping(estimator.get("initial_params"), field_name="estimator.initial_params")
            if solver in {"scipy_minimize", "transformed_scipy_minimize"}
            else None
        ),
        bounds=(
            _coerce_bounds_mapping(estimator.get("bounds"), field_name="estimator.bounds")
            if solver == "scipy_minimize"
            else None
        ),
        bounds_z=(
            _coerce_bounds_mapping(estimator.get("bounds_z"), field_name="estimator.bounds_z")
            if solver == "transformed_scipy_minimize"
            else None
        ),
        transforms=(
            _parse_transforms_mapping(estimator.get("transforms", {}), field_name="estimator.transforms")
            if solver == "transformed_scipy_minimize"
            else None
        ),
        method=str(estimator.get("method", "L-BFGS-B")),
        tol=float(estimator["tol"]) if "tol" in estimator else None,
        n_starts=int(estimator.get("n_starts", 5)),
        random_seed=(
            int(estimator["random_seed"])
            if "random_seed" in estimator and estimator["random_seed"] is not None
            else None
            if "random_seed" in estimator
            else 0
        ),
    )


def _validate_estimator_keys_for_solver(
    estimator: Mapping[str, Any],
    *,
    solver: MLESolverType,
    allow_solver_key: bool,
) -> None:
    """Validate estimator config keys for one MLE solver branch."""

    allowed: tuple[str, ...]
    if solver == "grid_search":
        allowed = ("type", "parameter_grid")
    elif solver == "scipy_minimize":
        allowed = ("type", "initial_params", "bounds", "method", "tol", "n_starts", "random_seed")
    elif solver == "transformed_scipy_minimize":
        allowed = (
            "type",
            "initial_params",
            "bounds_z",
            "transforms",
            "method",
            "tol",
            "n_starts",
            "random_seed",
        )
    else:
        raise ValueError(
            "solver must be one of "
            "{'grid_search', 'scipy_minimize', 'transformed_scipy_minimize'}"
        )

    if allow_solver_key:
        allowed = (*allowed, "solver")

    validate_allowed_keys(
        estimator,
        field_name="estimator",
        allowed_keys=allowed,
    )


def _coerce_fit_inference(raw: str, *, field_name: str) -> FitInferenceType:
    """Coerce a string to a supported fit inference literal."""

    if raw == "mle":
        return "mle"
    if raw == "bayesian":
        return "bayesian"
    raise ValueError(f"{field_name} must be one of {{'mle', 'bayesian'}}")


def _coerce_mle_solver(raw: str, *, field_name: str) -> MLESolverType:
    """Coerce a string to a supported MLE solver literal."""

    if raw == "grid_search":
        return "grid_search"
    if raw == "scipy_minimize":
        return "scipy_minimize"
    if raw == "transformed_scipy_minimize":
        return "transformed_scipy_minimize"
    raise ValueError(
        f"{field_name} must be one of "
        "{'grid_search', 'scipy_minimize', 'transformed_scipy_minimize'}"
    )


def model_component_spec_from_config(model_cfg: Mapping[str, Any]) -> ModelComponentSpec:
    """Parse model component spec from config mapping."""

    mapping = _require_mapping(model_cfg, field_name="model")
    validate_allowed_keys(mapping, field_name="model", allowed_keys=("component_id", "kwargs"))
    component_id = _coerce_non_empty_str(mapping.get("component_id"), field_name="model.component_id")
    kwargs = _require_mapping(mapping.get("kwargs", {}), field_name="model.kwargs")
    return ModelComponentSpec(component_id=component_id, kwargs=dict(kwargs))


def fit_dataset_from_config(
    data: EpisodeTrace | BlockData | tuple[TrialDecision, ...] | list[TrialDecision],
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
) -> MLEFitResult:
    """Fit a single dataset using declarative config.

    Parameters
    ----------
    data : EpisodeTrace | BlockData | tuple[TrialDecision, ...] | list[TrialDecision]
        Input dataset.
    config : Mapping[str, Any]
        Config mapping with keys ``model`` and ``estimator``.
    registry : PluginRegistry | None, optional
        Optional plugin registry.

    Returns
    -------
    MLEFitResult
        Fitting result.
    """

    cfg = _require_mapping(config, field_name="config")
    validate_allowed_keys(cfg, field_name="config", allowed_keys=("model", "estimator", "likelihood"))
    model_spec = model_component_spec_from_config(_require_mapping(cfg.get("model"), field_name="config.model"))
    fit_spec = fit_spec_from_config(_require_mapping(cfg.get("estimator"), field_name="config.estimator"))
    likelihood_cfg = (
        _require_mapping(cfg.get("likelihood"), field_name="config.likelihood")
        if "likelihood" in cfg
        else None
    )

    reg = registry if registry is not None else build_default_registry()
    return fit_model_from_registry(
        data,
        model_component_id=model_spec.component_id,
        fit_spec=fit_spec,
        model_kwargs=model_spec.kwargs,
        registry=reg,
        likelihood_program=likelihood_program_from_config(likelihood_cfg),
    )


def fit_block_from_config(
    block: BlockData,
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
) -> BlockFitResult:
    """Fit one block using declarative config."""

    cfg = _require_mapping(config, field_name="config")
    validate_allowed_keys(cfg, field_name="config", allowed_keys=("model", "estimator", "likelihood"))
    model_spec = model_component_spec_from_config(_require_mapping(cfg.get("model"), field_name="config.model"))
    fit_spec = fit_spec_from_config(_require_mapping(cfg.get("estimator"), field_name="config.estimator"))
    likelihood_cfg = (
        _require_mapping(cfg.get("likelihood"), field_name="config.likelihood")
        if "likelihood" in cfg
        else None
    )

    reg = registry if registry is not None else build_default_registry()
    return fit_block_data(
        block,
        model_component_id=model_spec.component_id,
        fit_spec=fit_spec,
        model_kwargs=model_spec.kwargs,
        registry=reg,
        likelihood_program=likelihood_program_from_config(likelihood_cfg),
    )


def fit_subject_from_config(
    subject: SubjectData,
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
) -> SubjectFitResult:
    """Fit one subject using declarative config."""

    cfg = _require_mapping(config, field_name="config")
    validate_allowed_keys(
        cfg,
        field_name="config",
        allowed_keys=("model", "estimator", "likelihood", "block_fit_strategy"),
    )
    model_spec = model_component_spec_from_config(_require_mapping(cfg.get("model"), field_name="config.model"))
    fit_spec = fit_spec_from_config(_require_mapping(cfg.get("estimator"), field_name="config.estimator"))
    likelihood_cfg = (
        _require_mapping(cfg.get("likelihood"), field_name="config.likelihood")
        if "likelihood" in cfg
        else None
    )
    block_fit_strategy: BlockFitStrategy = coerce_block_fit_strategy(
        cfg.get("block_fit_strategy"),
        field_name="config.block_fit_strategy",
    )

    reg = registry if registry is not None else build_default_registry()
    return fit_subject_data(
        subject,
        model_component_id=model_spec.component_id,
        fit_spec=fit_spec,
        model_kwargs=model_spec.kwargs,
        registry=reg,
        likelihood_program=likelihood_program_from_config(likelihood_cfg),
        block_fit_strategy=block_fit_strategy,
    )


def fit_study_from_config(
    study: StudyData,
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
) -> StudyFitResult:
    """Fit one study using declarative config."""

    cfg = _require_mapping(config, field_name="config")
    validate_allowed_keys(
        cfg,
        field_name="config",
        allowed_keys=("model", "estimator", "likelihood", "block_fit_strategy"),
    )
    model_spec = model_component_spec_from_config(_require_mapping(cfg.get("model"), field_name="config.model"))
    fit_spec = fit_spec_from_config(_require_mapping(cfg.get("estimator"), field_name="config.estimator"))
    likelihood_cfg = (
        _require_mapping(cfg.get("likelihood"), field_name="config.likelihood")
        if "likelihood" in cfg
        else None
    )
    block_fit_strategy: BlockFitStrategy = coerce_block_fit_strategy(
        cfg.get("block_fit_strategy"),
        field_name="config.block_fit_strategy",
    )

    reg = registry if registry is not None else build_default_registry()
    return fit_study_data(
        study,
        model_component_id=model_spec.component_id,
        fit_spec=fit_spec,
        model_kwargs=model_spec.kwargs,
        registry=reg,
        likelihood_program=likelihood_program_from_config(likelihood_cfg),
        block_fit_strategy=block_fit_strategy,
    )


def _parse_transforms_mapping(raw: Any, *, field_name: str) -> dict[str, ParameterTransform]:
    """Parse configured parameter transforms."""

    mapping = _require_mapping(raw, field_name=field_name)
    out: dict[str, ParameterTransform] = {}
    for param_name, spec in mapping.items():
        if isinstance(spec, str):
            out[str(param_name)] = _transform_from_name(spec)
            continue

        spec_mapping = _require_mapping(spec, field_name=f"{field_name}.{param_name}")
        kind = _coerce_non_empty_str(spec_mapping.get("kind"), field_name=f"{field_name}.{param_name}.kind")
        out[str(param_name)] = _transform_from_name(kind)
    return out


def _transform_from_name(name: str) -> ParameterTransform:
    """Resolve configured transform name."""

    normalized = str(name)
    if normalized == "identity":
        return identity_transform()
    if normalized == "unit_interval_logit":
        return unit_interval_logit_transform()
    if normalized == "positive_log":
        return positive_log_transform()
    raise ValueError(
        f"unsupported transform {name!r}; expected one of "
        "{'identity', 'unit_interval_logit', 'positive_log'}"
    )


def _coerce_bounds_mapping(raw: Any, *, field_name: str) -> dict[str, tuple[float | None, float | None]] | None:
    """Parse bounds mapping from config."""

    if raw is None:
        return None

    mapping = _require_mapping(raw, field_name=field_name)
    out: dict[str, tuple[float | None, float | None]] = {}
    for key, value in mapping.items():
        pair = _require_sequence(value, field_name=f"{field_name}.{key}")
        if len(pair) != 2:
            raise ValueError(f"{field_name}.{key} must have exactly two elements")
        lower_raw, upper_raw = pair
        lower = float(lower_raw) if lower_raw is not None else None
        upper = float(upper_raw) if upper_raw is not None else None
        out[str(key)] = (lower, upper)
    return out


def _coerce_float_list_mapping(raw: Any, *, field_name: str) -> dict[str, list[float]]:
    """Coerce mapping of parameter -> list[float]."""

    mapping = _require_mapping(raw, field_name=field_name)
    out: dict[str, list[float]] = {}
    for key, value in mapping.items():
        sequence = _require_sequence(value, field_name=f"{field_name}.{key}")
        out[str(key)] = [float(item) for item in sequence]
    return out


def _coerce_float_mapping(raw: Any, *, field_name: str) -> dict[str, float]:
    """Coerce mapping of parameter -> float."""

    mapping = _require_mapping(raw, field_name=field_name)
    return {str(key): float(value) for key, value in mapping.items()}


def _coerce_non_empty_str(raw: Any, *, field_name: str) -> str:
    """Coerce non-empty string with explicit field context."""

    if raw is None:
        raise ValueError(f"{field_name} must be a non-empty string")

    value = str(raw).strip()
    if not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_mapping(raw: Any, *, field_name: str) -> dict[str, Any]:
    """Require dictionary-like config value."""

    if not isinstance(raw, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return dict(raw)


def _require_sequence(raw: Any, *, field_name: str) -> list[Any]:
    """Require list-like config value."""

    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{field_name} must be an array")
    return list(raw)


__all__ = [
    "ModelComponentSpec",
    "fit_block_from_config",
    "fit_dataset_from_config",
    "fit_spec_from_config",
    "fit_study_from_config",
    "fit_subject_from_config",
    "model_component_spec_from_config",
]
