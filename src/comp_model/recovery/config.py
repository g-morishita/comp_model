"""Config-driven recovery workflow runners.

This module turns declarative mapping/JSON configs into executable recovery
runs using the plugin registry and inference fitting APIs.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from comp_model.generators import AsocialBlockSpec, SocialBlockSpec
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

    generating_ref = _parse_component_ref(_require_mapping(config, "generating_model"), field_name="generating_model")
    fitting_ref = _parse_component_ref(_require_mapping(config, "fitting_model"), field_name="fitting_model")
    estimator_cfg = _require_mapping(config, "estimator")
    prior_cfg = config.get("prior")
    likelihood_cfg = config.get("likelihood")

    true_parameter_sets = _parse_true_parameter_sets(config.get("true_parameter_sets"))
    n_trials = _coerce_positive_int(config.get("n_trials"), field_name="n_trials")
    seed = int(config.get("seed", 0))
    problem_factory, trace_factory = _build_simulation_sources(
        config=config,
        registry=reg,
        n_trials=n_trials,
    )

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
        problem_factory=problem_factory,
        model_factory=lambda params: reg.create_model(
            generating_ref.component_id,
            **_merge_kwargs(generating_ref.kwargs, params),
        ),
        fit_function=fit_function,
        true_parameter_sets=true_parameter_sets,
        n_trials=n_trials,
        seed=seed,
        trace_factory=trace_factory,
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
    problem_factory, trace_factory = _build_simulation_sources(
        config=config,
        registry=reg,
        n_trials=n_trials,
    )

    return run_model_recovery(
        problem_factory=problem_factory,
        generating_specs=tuple(generating_specs),
        candidate_specs=tuple(candidate_specs),
        n_trials=n_trials,
        n_replications_per_generator=n_replications,
        criterion=criterion,
        seed=seed,
        trace_factory=trace_factory,
    )


def _build_simulation_sources(
    *,
    config: dict[str, Any],
    registry: PluginRegistry,
    n_trials: int,
) -> tuple[Any, Any]:
    """Build problem/trace simulation sources from declarative config.

    Returns
    -------
    tuple[Any, Any]
        ``(problem_factory, trace_factory)`` pair. Exactly one entry is
        non-``None``.
    """

    simulation_cfg = (
        _require_mapping(config["simulation"], field_name="simulation")
        if "simulation" in config
        else None
    )
    if simulation_cfg is None:
        problem_ref = _parse_component_ref(
            _require_mapping(config, "problem"),
            field_name="problem",
        )
        return (
            lambda: registry.create_problem(problem_ref.component_id, **dict(problem_ref.kwargs)),
            None,
        )

    simulation_type = _coerce_non_empty_str(
        simulation_cfg.get("type", "problem"),
        field_name="simulation.type",
    )
    if simulation_type == "problem":
        if "problem" in simulation_cfg:
            problem_ref = _parse_component_ref(
                _require_mapping(simulation_cfg, "problem", field_name="simulation"),
                field_name="simulation.problem",
            )
        else:
            problem_ref = _parse_component_ref(
                _require_mapping(config, "problem"),
                field_name="problem",
            )
        return (
            lambda: registry.create_problem(problem_ref.component_id, **dict(problem_ref.kwargs)),
            None,
        )

    if simulation_type != "generator":
        raise ValueError("simulation.type must be one of {'problem', 'generator'}")

    generator_ref = _parse_component_ref(
        _require_mapping(simulation_cfg, "generator", field_name="simulation"),
        field_name="simulation.generator",
    )
    generator = registry.create_generator(
        generator_ref.component_id,
        **dict(generator_ref.kwargs),
    )

    block_cfg = _require_mapping(
        simulation_cfg.get("block", {}),
        field_name="simulation.block",
    )
    block_n_trials = int(block_cfg.get("n_trials", n_trials))
    if block_n_trials <= 0:
        raise ValueError("simulation.block.n_trials must be > 0")
    block_id = block_cfg.get("block_id")
    block_metadata = _require_mapping(
        block_cfg.get("metadata", {}),
        field_name="simulation.block.metadata",
    )

    signature = inspect.signature(generator.simulate_block)
    parameter_names = tuple(signature.parameters)

    if "model" in parameter_names:
        problem_kwargs = _require_mapping(
            block_cfg.get("problem_kwargs", {}),
            field_name="simulation.block.problem_kwargs",
        )

        def trace_factory(generating_model: Any, simulation_seed: int):
            block = AsocialBlockSpec(
                n_trials=block_n_trials,
                problem_kwargs=problem_kwargs,
                block_id=block_id,
                seed=int(simulation_seed),
                metadata=block_metadata,
            )
            block_data = generator.simulate_block(
                model=generating_model,
                block=block,
                rng=np.random.default_rng(simulation_seed),
            )
            trace = getattr(block_data, "event_trace", None)
            if trace is None:
                raise ValueError("generator outputs must include block.event_trace")
            return trace

        return None, trace_factory

    if "subject_model" in parameter_names and "demonstrator_model" in parameter_names:
        demonstrator_ref = _parse_component_ref(
            _require_mapping(simulation_cfg, "demonstrator_model", field_name="simulation"),
            field_name="simulation.demonstrator_model",
        )
        program_kwargs = _require_mapping(
            block_cfg.get("program_kwargs", {}),
            field_name="simulation.block.program_kwargs",
        )

        def trace_factory(generating_model: Any, simulation_seed: int):
            block = SocialBlockSpec(
                n_trials=block_n_trials,
                program_kwargs=program_kwargs,
                block_id=block_id,
                seed=int(simulation_seed),
                metadata=block_metadata,
            )
            demonstrator_model = registry.create_demonstrator(
                demonstrator_ref.component_id,
                **dict(demonstrator_ref.kwargs),
            )
            block_data = generator.simulate_block(
                subject_model=generating_model,
                demonstrator_model=demonstrator_model,
                block=block,
                rng=np.random.default_rng(simulation_seed),
            )
            trace = getattr(block_data, "event_trace", None)
            if trace is None:
                raise ValueError("generator outputs must include block.event_trace")
            return trace

        return None, trace_factory

    raise ValueError(
        "simulation.generator component must expose simulate_block with either "
        "('model', ...) or ('subject_model', 'demonstrator_model', ...)"
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
