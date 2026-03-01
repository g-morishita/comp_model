"""Config-driven recovery workflow runners.

This module turns declarative mapping/JSON configs into executable recovery
runs using the plugin registry and inference fitting APIs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from comp_model.inference.model_selection_config import build_fit_function_from_model_config
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
    """

    reg = registry if registry is not None else build_default_registry()

    problem_ref = _parse_component_ref(_require_mapping(config, "problem"), field_name="problem")
    generating_ref = _parse_component_ref(_require_mapping(config, "generating_model"), field_name="generating_model")
    fitting_ref = _parse_component_ref(_require_mapping(config, "fitting_model"), field_name="fitting_model")
    estimator_cfg = _require_mapping(config, "estimator")
    prior_cfg = config.get("prior")
    likelihood_cfg = config.get("likelihood")

    true_parameter_sets = _parse_true_parameter_sets(config.get("true_parameter_sets"))
    n_trials = _coerce_positive_int(config.get("n_trials"), field_name="n_trials")
    seed = int(config.get("seed", 0))

    fit_function = _build_fit_function(
        estimator_cfg=estimator_cfg,
        prior_cfg=prior_cfg,
        likelihood_cfg=(
            _require_mapping(likelihood_cfg, field_name="likelihood")
            if likelihood_cfg is not None
            else None
        ),
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
    global_likelihood_cfg = (
        _require_mapping(config["likelihood"], field_name="likelihood")
        if "likelihood" in config
        else None
    )

    generating_specs: list[GeneratingModelSpec] = []
    for index, raw in enumerate(generating_cfg):
        item = _require_mapping(raw, field_name=f"generating[{index}]")
        name = _coerce_non_empty_str(item.get("name"), field_name=f"generating[{index}].name")
        model_ref = _parse_component_ref(
            _require_mapping(item, "model", field_name=f"generating[{index}]"),
            field_name=f"generating[{index}].model",
        )
        true_params = _coerce_float_mapping(item.get("true_params", {}), field_name=f"generating[{index}].true_params")

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
        prior_cfg = item.get("prior")
        candidate_likelihood_cfg = (
            _require_mapping(item["likelihood"], field_name=f"candidates[{index}].likelihood")
            if "likelihood" in item
            else global_likelihood_cfg
        )

        fit_function = _build_fit_function(
            estimator_cfg=estimator_cfg,
            prior_cfg=prior_cfg,
            likelihood_cfg=candidate_likelihood_cfg,
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
    prior_cfg: Any,
    likelihood_cfg: dict[str, Any] | None,
    registry: PluginRegistry,
    fitting_ref: ComponentRef,
):
    """Build trace->fit callable from estimator and likelihood config."""

    model_cfg = {
        "component_id": fitting_ref.component_id,
        "kwargs": dict(fitting_ref.kwargs),
    }
    return build_fit_function_from_model_config(
        model_cfg=model_cfg,
        estimator_cfg=estimator_cfg,
        prior_cfg=(
            _require_mapping(prior_cfg, field_name="prior")
            if prior_cfg is not None
            else None
        ),
        likelihood_cfg=likelihood_cfg,
        registry=registry,
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
