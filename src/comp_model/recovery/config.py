"""Config-driven recovery workflow runners.

This module turns declarative mapping/JSON/YAML configs into executable recovery
runs using the plugin registry and inference fitting APIs.
"""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from comp_model.core import load_config_mapping
from comp_model.core.config_validation import validate_allowed_keys, validate_required_keys
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


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a recovery config file as a dictionary.

    Parameters
    ----------
    path : str | pathlib.Path
        Config file path (`.json`, `.yaml`, or `.yml`).

    Returns
    -------
    dict[str, Any]
        Parsed configuration mapping.
    """

    return load_config_mapping(path)


def load_json_config(path: str | Path) -> dict[str, Any]:
    """Backward-compatible alias for :func:`load_config`.

    Parameters
    ----------
    path : str | pathlib.Path
        Config file path (`.json`, `.yaml`, or `.yml`).

    Returns
    -------
    dict[str, Any]
        Parsed configuration mapping.
    """

    return load_config(path)


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
    validate_allowed_keys(
        config,
        field_name="config",
        allowed_keys=(
            "problem",
            "simulation",
            "generating_model",
            "fitting_model",
            "estimator",
            "prior",
            "likelihood",
            "true_parameter_sets",
            "n_trials",
            "seed",
        ),
    )
    validate_required_keys(
        config,
        field_name="config",
        required_keys=("generating_model", "fitting_model", "estimator", "true_parameter_sets", "n_trials"),
    )

    generating_ref = _parse_component_ref(_require_mapping(config, "generating_model"), field_name="generating_model")
    fitting_ref = _parse_component_ref(_require_mapping(config, "fitting_model"), field_name="fitting_model")
    estimator_cfg = _require_mapping(config, "estimator")
    prior_cfg = config.get("prior")
    likelihood_cfg = config.get("likelihood")

    true_parameter_sets = _parse_true_parameter_sets(config.get("true_parameter_sets"))
    n_trials = _coerce_positive_int(config.get("n_trials"), field_name="n_trials")
    seed = int(config.get("seed", 0))
    problem_factory, trace_factory, simulation_level = _build_simulation_sources(
        config=config,
        registry=reg,
        n_trials=n_trials,
    )
    if simulation_level != "block":
        raise ValueError(
            "parameter recovery currently supports simulation.level='block' only"
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
    validate_allowed_keys(
        config,
        field_name="config",
        allowed_keys=(
            "problem",
            "simulation",
            "generating",
            "candidates",
            "likelihood",
            "n_trials",
            "n_replications_per_generator",
            "criterion",
            "seed",
        ),
    )
    validate_required_keys(
        config,
        field_name="config",
        required_keys=("generating", "candidates", "n_trials", "n_replications_per_generator"),
    )

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
    problem_factory, trace_factory, _ = _build_simulation_sources(
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
) -> tuple[Any, Any, str]:
    """Build problem/trace simulation sources from declarative config.

    Returns
    -------
    tuple[Any, Any, str]
        ``(problem_factory, trace_factory, simulation_level)`` where exactly
        one of ``problem_factory`` and ``trace_factory`` is non-``None``.
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
            "block",
        )

    simulation_type = _coerce_non_empty_str(
        simulation_cfg.get("type", "problem"),
        field_name="simulation.type",
    )
    simulation_level = _coerce_non_empty_str(
        simulation_cfg.get("level", "block"),
        field_name="simulation.level",
    )
    if simulation_level not in {"block", "subject", "study"}:
        raise ValueError("simulation.level must be one of {'block', 'subject', 'study'}")

    if simulation_type == "problem":
        validate_allowed_keys(
            simulation_cfg,
            field_name="simulation",
            allowed_keys=("type", "problem", "level"),
        )
        if simulation_level != "block":
            raise ValueError(
                "simulation.level must be 'block' when simulation.type='problem'"
            )
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
            "block",
        )

    if simulation_type != "generator":
        raise ValueError("simulation.type must be one of {'problem', 'generator'}")
    validate_allowed_keys(
        simulation_cfg,
        field_name="simulation",
        allowed_keys=(
            "type",
            "level",
            "generator",
            "demonstrator_model",
            "block",
            "blocks",
            "subject_id",
            "subject_ids",
            "n_subjects",
        ),
    )

    generator_ref = _parse_component_ref(
        _require_mapping(simulation_cfg, "generator", field_name="simulation"),
        field_name="simulation.generator",
    )
    generator = registry.create_generator(
        generator_ref.component_id,
        **dict(generator_ref.kwargs),
    )

    if "block" in simulation_cfg and "blocks" in simulation_cfg:
        raise ValueError("simulation must include either 'block' or 'blocks', not both")
    block_rows = (
        _require_sequence(simulation_cfg["blocks"], field_name="simulation.blocks")
        if "blocks" in simulation_cfg
        else [simulation_cfg.get("block", {})]
    )
    if not block_rows:
        raise ValueError("simulation.blocks must include at least one block spec")
    if simulation_level == "block" and len(block_rows) != 1:
        raise ValueError("simulation.level='block' requires exactly one block spec")

    subject_id = _coerce_non_empty_str(
        simulation_cfg.get("subject_id", "subject"),
        field_name="simulation.subject_id",
    )
    if "subject_ids" in simulation_cfg:
        raw_subject_ids = _require_sequence(simulation_cfg["subject_ids"], field_name="simulation.subject_ids")
        subject_ids = tuple(
            _coerce_non_empty_str(value, field_name=f"simulation.subject_ids[{index}]")
            for index, value in enumerate(raw_subject_ids)
        )
        if not subject_ids:
            raise ValueError("simulation.subject_ids must include at least one subject")
    else:
        n_subjects = int(simulation_cfg.get("n_subjects", 1))
        if n_subjects <= 0:
            raise ValueError("simulation.n_subjects must be > 0")
        subject_ids = tuple(f"s{index+1}" for index in range(n_subjects))

    signature = inspect.signature(generator.simulate_block)
    parameter_names = tuple(signature.parameters)

    if "model" in parameter_names:
        blocks: list[AsocialBlockSpec] = []
        for index, raw_block in enumerate(block_rows):
            block_cfg = _require_mapping(raw_block, field_name=f"simulation.blocks[{index}]")
            validate_allowed_keys(
                block_cfg,
                field_name=f"simulation.blocks[{index}]",
                allowed_keys=("n_trials", "block_id", "metadata", "problem_kwargs"),
            )
            block_n_trials = int(block_cfg.get("n_trials", n_trials))
            if block_n_trials <= 0:
                raise ValueError(f"simulation.blocks[{index}].n_trials must be > 0")
            blocks.append(
                AsocialBlockSpec(
                    n_trials=block_n_trials,
                    problem_kwargs=_require_mapping(
                        block_cfg.get("problem_kwargs", {}),
                        field_name=f"simulation.blocks[{index}].problem_kwargs",
                    ),
                    block_id=block_cfg.get("block_id", f"b{index}"),
                    seed=None,
                    metadata=_require_mapping(
                        block_cfg.get("metadata", {}),
                        field_name=f"simulation.blocks[{index}].metadata",
                    ),
                )
            )
        block_specs = tuple(blocks)

        def trace_factory(generating_model: Any, simulation_seed: int):
            rng = np.random.default_rng(simulation_seed)
            if simulation_level == "block":
                block = copy.copy(block_specs[0])
                block = AsocialBlockSpec(
                    n_trials=block.n_trials,
                    problem_kwargs=block.problem_kwargs,
                    block_id=block.block_id,
                    seed=int(simulation_seed),
                    metadata=block.metadata,
                )
                block_data = generator.simulate_block(
                    model=generating_model,
                    block=block,
                    rng=rng,
                )
                trace = getattr(block_data, "event_trace", None)
                if trace is None:
                    raise ValueError("generator outputs must include block.event_trace")
                return trace
            if simulation_level == "subject":
                return generator.simulate_subject(
                    subject_id=subject_id,
                    model=generating_model,
                    blocks=block_specs,
                    rng=rng,
                )

            model_by_subject = {
                sid: (generating_model if index == 0 else copy.deepcopy(generating_model))
                for index, sid in enumerate(subject_ids)
            }
            return generator.simulate_study(
                subject_models=model_by_subject,
                blocks=block_specs,
                rng=rng,
            )

        return None, trace_factory, simulation_level

    if "subject_model" in parameter_names and "demonstrator_model" in parameter_names:
        demonstrator_ref = _parse_component_ref(
            _require_mapping(simulation_cfg, "demonstrator_model", field_name="simulation"),
            field_name="simulation.demonstrator_model",
        )
        blocks: list[SocialBlockSpec] = []
        for index, raw_block in enumerate(block_rows):
            block_cfg = _require_mapping(raw_block, field_name=f"simulation.blocks[{index}]")
            validate_allowed_keys(
                block_cfg,
                field_name=f"simulation.blocks[{index}]",
                allowed_keys=("n_trials", "block_id", "metadata", "program_kwargs"),
            )
            block_n_trials = int(block_cfg.get("n_trials", n_trials))
            if block_n_trials <= 0:
                raise ValueError(f"simulation.blocks[{index}].n_trials must be > 0")
            blocks.append(
                SocialBlockSpec(
                    n_trials=block_n_trials,
                    program_kwargs=_require_mapping(
                        block_cfg.get("program_kwargs", {}),
                        field_name=f"simulation.blocks[{index}].program_kwargs",
                    ),
                    block_id=block_cfg.get("block_id", f"b{index}"),
                    seed=None,
                    metadata=_require_mapping(
                        block_cfg.get("metadata", {}),
                        field_name=f"simulation.blocks[{index}].metadata",
                    ),
                )
            )
        block_specs = tuple(blocks)

        def trace_factory(generating_model: Any, simulation_seed: int):
            rng = np.random.default_rng(simulation_seed)

            def make_demonstrator():
                return registry.create_demonstrator(
                    demonstrator_ref.component_id,
                    **dict(demonstrator_ref.kwargs),
                )

            if simulation_level == "block":
                block = copy.copy(block_specs[0])
                block = SocialBlockSpec(
                    n_trials=block.n_trials,
                    program_kwargs=block.program_kwargs,
                    block_id=block.block_id,
                    seed=int(simulation_seed),
                    metadata=block.metadata,
                )
                block_data = generator.simulate_block(
                    subject_model=generating_model,
                    demonstrator_model=make_demonstrator(),
                    block=block,
                    rng=rng,
                )
                trace = getattr(block_data, "event_trace", None)
                if trace is None:
                    raise ValueError("generator outputs must include block.event_trace")
                return trace
            if simulation_level == "subject":
                return generator.simulate_subject(
                    subject_id=subject_id,
                    subject_model=generating_model,
                    demonstrator_model=make_demonstrator(),
                    blocks=block_specs,
                    rng=rng,
                )

            subject_models = {
                sid: (generating_model if index == 0 else copy.deepcopy(generating_model))
                for index, sid in enumerate(subject_ids)
            }
            demonstrator_models = {sid: make_demonstrator() for sid in subject_ids}
            return generator.simulate_study(
                subject_models=subject_models,
                demonstrator_models=demonstrator_models,
                blocks=block_specs,
                rng=rng,
            )

        return None, trace_factory, simulation_level

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
    "load_config",
    "load_json_config",
    "run_model_recovery_from_config",
    "run_parameter_recovery_from_config",
]
