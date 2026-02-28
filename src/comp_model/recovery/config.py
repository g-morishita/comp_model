"""Config-driven recovery workflow runners.

This module turns declarative mapping/JSON configs into executable recovery
runs using the plugin registry and inference estimators.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from comp_model.inference import (
    ActionReplayLikelihood,
    GridSearchMLEEstimator,
    MLEFitResult,
    ScipyMinimizeMLEEstimator,
    TransformedScipyMinimizeMLEEstimator,
    identity_transform,
    positive_log_transform,
    unit_interval_logit_transform,
)
from comp_model.plugins import PluginRegistry, build_default_registry
from comp_model.recovery.model import CandidateModelSpec, GeneratingModelSpec, ModelRecoveryResult, run_model_recovery
from comp_model.recovery.parameter import ParameterRecoveryResult, run_parameter_recovery


@dataclass(frozen=True, slots=True)
class ComponentRef:
    """Registry component reference.

    Parameters
    ----------
    component_id : str
        Component ID in the plugin registry.
    kwargs : dict[str, Any]
        Fixed constructor keyword arguments applied on creation.
    """

    component_id: str
    kwargs: dict[str, Any]


def load_json_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON config file as a dictionary.

    Parameters
    ----------
    path : str | pathlib.Path
        JSON config file path.

    Returns
    -------
    dict[str, Any]
        Parsed configuration mapping.
    """

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, dict):
        raise ValueError("config root must be a JSON object")
    return raw


def run_parameter_recovery_from_config(
    config: dict[str, Any],
    *,
    registry: PluginRegistry | None = None,
) -> ParameterRecoveryResult:
    """Run parameter recovery from declarative configuration.

    Parameters
    ----------
    config : dict[str, Any]
        Parameter-recovery configuration mapping.
    registry : PluginRegistry | None, optional
        Optional pre-built registry. Defaults to built-in registry.

    Returns
    -------
    ParameterRecoveryResult
        Recovery result.

    Notes
    -----
    Required config fields:
    - ``problem`` component reference
    - ``generating_model`` component reference
    - ``fitting_model`` component reference
    - ``estimator`` definition
    - ``true_parameter_sets`` list
    - ``n_trials`` integer
    """

    reg = registry if registry is not None else build_default_registry()

    problem_ref = _parse_component_ref(_require_mapping(config, "problem"), field_name="problem")
    generating_ref = _parse_component_ref(
        _require_mapping(config, "generating_model"),
        field_name="generating_model",
    )
    fitting_ref = _parse_component_ref(
        _require_mapping(config, "fitting_model"),
        field_name="fitting_model",
    )
    estimator_cfg = _require_mapping(config, "estimator")

    true_parameter_sets = _parse_true_parameter_sets(config.get("true_parameter_sets"))
    n_trials = _coerce_positive_int(config.get("n_trials"), field_name="n_trials")
    seed = int(config.get("seed", 0))

    fit_function = _build_fit_function(
        estimator_cfg=estimator_cfg,
        registry=reg,
        fitting_ref=fitting_ref,
    )

    return run_parameter_recovery(
        problem_factory=lambda: reg.create_problem(problem_ref.component_id, **dict(problem_ref.kwargs)),
        model_factory=lambda params: reg.create_model(
            generating_ref.component_id,
            **_merge_kwargs(generating_ref.kwargs, params),
        ),
        fit_function=fit_function,
        true_parameter_sets=true_parameter_sets,
        n_trials=n_trials,
        seed=seed,
    )


def run_model_recovery_from_config(
    config: dict[str, Any],
    *,
    registry: PluginRegistry | None = None,
) -> ModelRecoveryResult:
    """Run model recovery from declarative configuration.

    Parameters
    ----------
    config : dict[str, Any]
        Model-recovery configuration mapping.
    registry : PluginRegistry | None, optional
        Optional pre-built registry. Defaults to built-in registry.

    Returns
    -------
    ModelRecoveryResult
        Recovery result.
    """

    reg = registry if registry is not None else build_default_registry()

    problem_ref = _parse_component_ref(_require_mapping(config, "problem"), field_name="problem")
    generating_cfg = _require_sequence(config.get("generating"), field_name="generating")
    candidate_cfg = _require_sequence(config.get("candidates"), field_name="candidates")

    generating_specs: list[GeneratingModelSpec] = []
    for index, raw in enumerate(generating_cfg):
        item = _require_mapping(raw, field_name=f"generating[{index}]")
        name = _coerce_non_empty_str(item.get("name"), field_name=f"generating[{index}].name")
        model_ref = _parse_component_ref(
            _require_mapping(item, "model", field_name=f"generating[{index}]"),
            field_name=f"generating[{index}].model",
        )
        true_params = _coerce_float_mapping(
            item.get("true_params", {}),
            field_name=f"generating[{index}].true_params",
        )

        generating_specs.append(
            GeneratingModelSpec(
                name=name,
                model_factory=lambda params, model_ref=model_ref: reg.create_model(
                    model_ref.component_id,
                    **_merge_kwargs(model_ref.kwargs, params),
                ),
                true_params=true_params,
            )
        )

    candidate_specs: list[CandidateModelSpec] = []
    for index, raw in enumerate(candidate_cfg):
        item = _require_mapping(raw, field_name=f"candidates[{index}]")
        name = _coerce_non_empty_str(item.get("name"), field_name=f"candidates[{index}].name")
        model_ref = _parse_component_ref(
            _require_mapping(item, "model", field_name=f"candidates[{index}]"),
            field_name=f"candidates[{index}].model",
        )
        estimator_cfg = _require_mapping(item, "estimator", field_name=f"candidates[{index}]")

        fit_function = _build_fit_function(
            estimator_cfg=estimator_cfg,
            registry=reg,
            fitting_ref=model_ref,
        )

        n_parameters_raw = item.get("n_parameters")
        n_parameters = int(n_parameters_raw) if n_parameters_raw is not None else None

        candidate_specs.append(
            CandidateModelSpec(
                name=name,
                fit_function=fit_function,
                n_parameters=n_parameters,
            )
        )

    n_trials = _coerce_positive_int(config.get("n_trials"), field_name="n_trials")
    n_replications = _coerce_positive_int(
        config.get("n_replications_per_generator"),
        field_name="n_replications_per_generator",
    )
    criterion = str(config.get("criterion", "log_likelihood"))
    seed = int(config.get("seed", 0))

    return run_model_recovery(
        problem_factory=lambda: reg.create_problem(problem_ref.component_id, **dict(problem_ref.kwargs)),
        generating_specs=tuple(generating_specs),
        candidate_specs=tuple(candidate_specs),
        n_trials=n_trials,
        n_replications_per_generator=n_replications,
        criterion=criterion,
        seed=seed,
    )


def _build_fit_function(
    *,
    estimator_cfg: dict[str, Any],
    registry: PluginRegistry,
    fitting_ref: ComponentRef,
):
    """Build trace->MLE fit callable from estimator config."""

    estimator_type = _coerce_non_empty_str(estimator_cfg.get("type"), field_name="estimator.type")

    model_manifest = registry.get("model", fitting_ref.component_id)
    likelihood_program = ActionReplayLikelihood()

    model_factory = lambda params: registry.create_model(
        fitting_ref.component_id,
        **_merge_kwargs(fitting_ref.kwargs, params),
    )

    if estimator_type == "grid_search":
        parameter_grid = _coerce_float_list_mapping(
            estimator_cfg.get("parameter_grid"),
            field_name="estimator.parameter_grid",
        )
        estimator = GridSearchMLEEstimator(
            likelihood_program=likelihood_program,
            model_factory=model_factory,
            requirements=model_manifest.requirements,
        )
        return lambda trace: estimator.fit(trace=trace, parameter_grid=parameter_grid)

    if estimator_type == "scipy_minimize":
        initial_params = _coerce_float_mapping(
            estimator_cfg.get("initial_params"),
            field_name="estimator.initial_params",
        )
        bounds = _coerce_bounds_mapping(estimator_cfg.get("bounds"), field_name="estimator.bounds")
        estimator = ScipyMinimizeMLEEstimator(
            likelihood_program=likelihood_program,
            model_factory=model_factory,
            requirements=model_manifest.requirements,
            method=str(estimator_cfg.get("method", "L-BFGS-B")),
            tol=float(estimator_cfg["tol"]) if "tol" in estimator_cfg else None,
        )
        return lambda trace: estimator.fit(trace=trace, initial_params=initial_params, bounds=bounds)

    if estimator_type == "transformed_scipy_minimize":
        initial_params = _coerce_float_mapping(
            estimator_cfg.get("initial_params"),
            field_name="estimator.initial_params",
        )
        bounds_z = _coerce_bounds_mapping(estimator_cfg.get("bounds_z"), field_name="estimator.bounds_z")
        transforms = _parse_transforms_mapping(
            estimator_cfg.get("transforms", {}),
            field_name="estimator.transforms",
        )

        estimator = TransformedScipyMinimizeMLEEstimator(
            likelihood_program=likelihood_program,
            model_factory=model_factory,
            transforms=transforms,
            requirements=model_manifest.requirements,
            method=str(estimator_cfg.get("method", "L-BFGS-B")),
            tol=float(estimator_cfg["tol"]) if "tol" in estimator_cfg else None,
        )
        return lambda trace: estimator.fit(trace=trace, initial_params=initial_params, bounds_z=bounds_z)

    raise ValueError(
        "estimator.type must be one of "
        "{'grid_search', 'scipy_minimize', 'transformed_scipy_minimize'}"
    )


def _parse_component_ref(raw: dict[str, Any], *, field_name: str) -> ComponentRef:
    """Parse one component reference mapping."""

    component_id = _coerce_non_empty_str(raw.get("component_id"), field_name=f"{field_name}.component_id")
    kwargs = _require_mapping(raw.get("kwargs", {}), field_name=f"{field_name}.kwargs")
    return ComponentRef(component_id=component_id, kwargs=dict(kwargs))


def _parse_true_parameter_sets(raw: Any) -> tuple[dict[str, float], ...]:
    """Parse list of true parameter dictionaries."""

    seq = _require_sequence(raw, field_name="true_parameter_sets")
    out: list[dict[str, float]] = []
    for index, item in enumerate(seq):
        out.append(_coerce_float_mapping(item, field_name=f"true_parameter_sets[{index}]"))
    return tuple(out)


def _parse_transforms_mapping(raw: Any, *, field_name: str) -> dict[str, Any]:
    """Parse parameter-transform mapping from config."""

    mapping = _require_mapping(raw, field_name=field_name)
    out: dict[str, Any] = {}
    for param_name, spec in mapping.items():
        if isinstance(spec, str):
            out[str(param_name)] = _transform_from_name(spec)
            continue

        spec_mapping = _require_mapping(spec, field_name=f"{field_name}.{param_name}")
        kind = _coerce_non_empty_str(spec_mapping.get("kind"), field_name=f"{field_name}.{param_name}.kind")
        out[str(param_name)] = _transform_from_name(kind)
    return out


def _transform_from_name(name: str):
    """Resolve configured transform name to transform instance."""

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


def _coerce_positive_int(raw: Any, *, field_name: str) -> int:
    """Coerce positive integer from arbitrary config scalar."""

    if raw is None:
        raise ValueError(f"{field_name} is required")

    value = int(raw)
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return value


def _coerce_non_empty_str(raw: Any, *, field_name: str) -> str:
    """Coerce non-empty string with explicit field context."""

    if raw is None:
        raise ValueError(f"{field_name} must be a non-empty string")

    value = str(raw).strip()
    if not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_mapping(raw: Any, key: str | None = None, *, field_name: str | None = None) -> dict[str, Any]:
    """Require dictionary-like value and optionally nested key."""

    if key is not None:
        if not isinstance(raw, dict):
            label = field_name or "value"
            raise ValueError(f"{label} must be an object")
        if key not in raw:
            label = field_name or "value"
            raise ValueError(f"{label}.{key} is required")
        raw = raw[key]

    if not isinstance(raw, dict):
        label = field_name or "value"
        raise ValueError(f"{label} must be an object")
    return raw


def _require_sequence(raw: Any, *, field_name: str) -> list[Any]:
    """Require list-like value in config."""

    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{field_name} must be an array")
    return list(raw)


def _merge_kwargs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge constructor kwargs with explicit parameter overrides."""

    merged = dict(base)
    merged.update(override)
    return merged


__all__ = [
    "ComponentRef",
    "load_json_config",
    "run_model_recovery_from_config",
    "run_parameter_recovery_from_config",
]
