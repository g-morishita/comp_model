"""Config-driven recovery workflow runners.

This module turns declarative mapping/JSON/YAML configs into executable recovery
runs using the plugin registry and inference fitting APIs.
"""

from __future__ import annotations

import copy
import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from comp_model.core import load_config_mapping
from comp_model.core.config_validation import validate_allowed_keys, validate_required_keys
from comp_model.core.data import SubjectData
from comp_model.generators import AsocialBlockSpec, SocialBlockSpec
from comp_model.inference.block_strategy import BlockFitStrategy, coerce_block_fit_strategy
from comp_model.inference.estimator_dispatch import (
    BAYES_ESTIMATORS,
    MLE_ESTIMATORS,
    fit_study_auto_from_config,
    fit_subject_auto_from_config,
)
from comp_model.inference.mle.estimators import MLECandidate, MLEFitResult
from comp_model.inference.mle.group import SubjectFitResult
from comp_model.inference.model_selection import SelectionCriterion
from comp_model.inference.model_selection_config import build_fit_function_from_model_config
from comp_model.inference.transforms import (
    ParameterTransform,
    identity_transform,
    positive_log_transform,
    unit_interval_logit_transform,
)
from comp_model.plugins import PluginRegistry, build_default_registry
from comp_model.recovery.model import (
    CandidateModelSpec,
    GeneratingModelSpec,
    ModelRecoveryResult,
    run_model_recovery,
)
from comp_model.recovery.parameter import (
    ParameterRecoveryResult,
    run_parameter_recovery,
)
from comp_model.recovery.parameter import (
    resolve_true_parameter_sets as resolve_true_parameter_sets_api,
)


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
            "block_fit_strategy",
            "generating_model",
            "fitting_model",
            "estimator",
            "prior",
            "likelihood",
            "true_parameter_sets",
            "true_parameter_distributions",
            "sampling",
            "n_parameter_sets",
            "n_trials",
            "seed",
        ),
    )
    validate_required_keys(
        config,
        field_name="config",
        required_keys=("generating_model", "fitting_model", "estimator", "n_trials"),
    )

    generating_ref = _parse_component_ref(_require_mapping(config, "generating_model"), field_name="generating_model")
    fitting_ref = _parse_component_ref(_require_mapping(config, "fitting_model"), field_name="fitting_model")
    estimator_cfg = _require_mapping(config, "estimator")
    prior_cfg = config.get("prior")
    likelihood_cfg = config.get("likelihood")

    n_trials = _coerce_positive_int(config.get("n_trials"), field_name="n_trials")
    seed = int(config.get("seed", 0))
    block_fit_strategy: BlockFitStrategy = coerce_block_fit_strategy(
        config.get("block_fit_strategy"),
        field_name="config.block_fit_strategy",
    )
    true_parameter_sets = _resolve_true_parameter_sets(config, seed=seed)
    problem_factory, trace_factory, simulation_level = _build_simulation_sources(
        config=config,
        registry=reg,
        n_trials=n_trials,
    )
    if simulation_level not in {"block", "subject"}:
        raise ValueError(
            "parameter recovery currently supports simulation.level in {'block', 'subject'}"
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
        simulation_level=simulation_level,
        block_fit_strategy=block_fit_strategy,
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
            "block_fit_strategy",
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
    block_fit_strategy: BlockFitStrategy = coerce_block_fit_strategy(
        config.get("block_fit_strategy"),
        field_name="config.block_fit_strategy",
    )
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

        def model_factory(
            params: dict[str, float],
            model_ref: ComponentRef = model_ref,
        ) -> Any:
            return reg.create_model(
                model_ref.component_id,
                **_merge_kwargs(model_ref.kwargs, params),
            )

        generating_specs.append(
            GeneratingModelSpec(
                name=name,
                model_factory=model_factory,
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
            simulation_level="block",
            block_fit_strategy="independent",
        )
        candidate_fit_config: dict[str, Any] = {
            "model": {
                "component_id": model_ref.component_id,
                "kwargs": dict(model_ref.kwargs),
            },
            "estimator": dict(estimator_cfg),
            "block_fit_strategy": block_fit_strategy,
        }
        if prior_cfg is not None:
            candidate_fit_config["prior"] = dict(
                _require_mapping(prior_cfg, field_name=f"candidates[{index}].prior")
            )
        if candidate_likelihood_cfg is not None:
            candidate_fit_config["likelihood"] = dict(candidate_likelihood_cfg)

        n_parameters_raw = item.get("n_parameters")
        n_parameters = int(n_parameters_raw) if n_parameters_raw is not None else None

        def fit_subject_from_candidate(
            subject: Any,
            *,
            _config: dict[str, Any] = candidate_fit_config,
            _registry: PluginRegistry = reg,
        ) -> Any:
            return fit_subject_auto_from_config(
                subject,
                config=_config,
                registry=_registry,
            )

        def fit_study_from_candidate(
            study: Any,
            *,
            _config: dict[str, Any] = candidate_fit_config,
            _registry: PluginRegistry = reg,
        ) -> Any:
            return fit_study_auto_from_config(
                study,
                config=_config,
                registry=_registry,
            )

        candidate_specs.append(
            CandidateModelSpec(
                name=name,
                fit_function=fit_function,
                n_parameters=n_parameters,
                fit_subject_function=fit_subject_from_candidate,
                fit_study_function=fit_study_from_candidate,
            )
        )

    n_trials = _coerce_positive_int(config.get("n_trials"), field_name="n_trials")
    n_replications = _coerce_positive_int(
        config.get("n_replications_per_generator"),
        field_name="n_replications_per_generator",
    )
    criterion = _parse_selection_criterion(
        config.get("criterion", "log_likelihood"),
        field_name="criterion",
    )
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
        block_fit_strategy=block_fit_strategy,
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
        asocial_blocks: list[AsocialBlockSpec] = []
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
            asocial_blocks.append(
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
        block_specs = tuple(asocial_blocks)

        def trace_factory(generating_model: Any, simulation_seed: int):
            rng = np.random.default_rng(simulation_seed)
            if simulation_level == "block":
                block_spec = copy.copy(block_specs[0])
                block_spec = AsocialBlockSpec(
                    n_trials=block_spec.n_trials,
                    problem_kwargs=block_spec.problem_kwargs,
                    block_id=block_spec.block_id,
                    seed=int(simulation_seed),
                    metadata=block_spec.metadata,
                )
                block_data = generator.simulate_block(
                    model=generating_model,
                    block=block_spec,
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
        social_blocks: list[SocialBlockSpec] = []
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
            social_blocks.append(
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
        social_block_specs = tuple(social_blocks)

        def trace_factory(generating_model: Any, simulation_seed: int):
            rng = np.random.default_rng(simulation_seed)

            def make_demonstrator():
                return registry.create_demonstrator(
                    demonstrator_ref.component_id,
                    **dict(demonstrator_ref.kwargs),
                )

            if simulation_level == "block":
                block_spec = copy.copy(social_block_specs[0])
                block_spec = SocialBlockSpec(
                    n_trials=block_spec.n_trials,
                    program_kwargs=block_spec.program_kwargs,
                    block_id=block_spec.block_id,
                    seed=int(simulation_seed),
                    metadata=block_spec.metadata,
                )
                block_data = generator.simulate_block(
                    subject_model=generating_model,
                    demonstrator_model=make_demonstrator(),
                    block=block_spec,
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
                    blocks=social_block_specs,
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
                blocks=social_block_specs,
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
    simulation_level: str,
    block_fit_strategy: BlockFitStrategy,
):
    """Build trace->fit callable from estimator and likelihood config."""

    model_cfg = {
        "component_id": fitting_ref.component_id,
        "kwargs": dict(fitting_ref.kwargs),
    }
    if simulation_level == "block":
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

    estimator_type = _coerce_non_empty_str(estimator_cfg.get("type"), field_name="estimator.type")
    fit_config: dict[str, Any] = {
        "model": {
            "component_id": fitting_ref.component_id,
            "kwargs": dict(fitting_ref.kwargs),
        },
        "estimator": dict(estimator_cfg),
    }

    if estimator_type in MLE_ESTIMATORS:
        if likelihood_cfg is not None:
            fit_config["likelihood"] = dict(likelihood_cfg)
        fit_config["block_fit_strategy"] = block_fit_strategy
    elif estimator_type in BAYES_ESTIMATORS:
        if prior_cfg is not None:
            raise ValueError(
                f"prior is not supported for estimator type {estimator_type!r}"
            )
        if likelihood_cfg is not None:
            raise ValueError(
                f"likelihood is not supported for estimator type {estimator_type!r}"
            )
    else:
        supported = sorted(
            MLE_ESTIMATORS | BAYES_ESTIMATORS
        )
        raise ValueError(
            f"estimator.type must be one of {supported}; got {estimator_type!r}"
        )

    def _fit_subject(data: Any) -> Any:
        if not isinstance(data, SubjectData):
            raise ValueError("subject-level parameter recovery requires SubjectData traces")
        result = fit_subject_auto_from_config(
            data,
            config=fit_config,
            registry=registry,
        )
        if isinstance(result, SubjectFitResult):
            if result.shared_best_params is None:
                raise ValueError(
                    "subject-level parameter recovery requires one shared parameter "
                    "estimate per subject; set block_fit_strategy='joint' or use "
                    "block-level recovery"
                )
            if isinstance(result.shared_best_params, Mapping):
                candidate = MLECandidate(
                    params={str(k): float(v) for k, v in result.shared_best_params.items()},
                    log_likelihood=float(result.total_log_likelihood),
                )
                return MLEFitResult(best=candidate, candidates=(candidate,))
        return result

    return _fit_subject


def _parse_component_ref(raw: dict[str, Any], *, field_name: str) -> ComponentRef:
    """Parse one component reference mapping."""

    component_id = _coerce_non_empty_str(raw.get("component_id"), field_name=f"{field_name}.component_id")
    kwargs = _require_mapping(raw.get("kwargs", {}), field_name=f"{field_name}.kwargs")
    return ComponentRef(component_id=component_id, kwargs=dict(kwargs))


def _resolve_true_parameter_sets(
    config: dict[str, Any],
    *,
    seed: int,
) -> tuple[dict[str, float], ...]:
    """Resolve true parameters from explicit sets or sampled distributions."""

    has_sets = "true_parameter_sets" in config
    has_distributions = "true_parameter_distributions" in config
    has_sampling = "sampling" in config

    selected_count = int(has_sets) + int(has_distributions) + int(has_sampling)
    if selected_count > 1:
        raise ValueError(
            "config must include exactly one of "
            "{'true_parameter_sets', 'true_parameter_distributions', 'sampling'}"
        )
    if selected_count == 0:
        raise ValueError(
            "config.true_parameter_sets, config.true_parameter_distributions, "
            "or config.sampling is required"
        )

    if has_sampling:
        sampling_cfg = _require_mapping(config.get("sampling"), field_name="sampling")
        n_parameter_sets = _coerce_positive_int(
            sampling_cfg.get("n_parameter_sets", config.get("n_parameter_sets")),
            field_name="sampling.n_parameter_sets",
        )
        return _sample_true_parameter_sets_from_sampling(
            sampling_cfg,
            n_parameter_sets=n_parameter_sets,
            seed=seed,
        )

    return resolve_true_parameter_sets_api(
        true_parameter_sets=(
            _require_sequence(config.get("true_parameter_sets"), field_name="true_parameter_sets")
            if has_sets
            else None
        ),
        true_parameter_distributions=(
            _require_mapping(
                config.get("true_parameter_distributions"),
                field_name="true_parameter_distributions",
            )
            if has_distributions
            else None
        ),
        n_parameter_sets=(
            _coerce_positive_int(
                config.get("n_parameter_sets"),
                field_name="n_parameter_sets",
            )
            if has_distributions
            else None
        ),
        seed=seed,
    )


def _sample_true_parameter_sets_from_sampling(
    sampling_cfg: dict[str, Any],
    *,
    n_parameter_sets: int,
    seed: int,
) -> tuple[dict[str, float], ...]:
    """Sample true parameter dictionaries from the ``sampling`` config section."""

    validate_allowed_keys(
        sampling_cfg,
        field_name="sampling",
        allowed_keys=(
            "mode",
            "space",
            "n_parameter_sets",
            "distributions",
            "population",
            "individual_sd",
            "transforms",
            "bounds",
            "clip_to_bounds",
            "by_condition",
            "conditions",
            "baseline_condition",
        ),
    )
    mode = _coerce_non_empty_str(sampling_cfg.get("mode", "independent"), field_name="sampling.mode")
    if mode == "fixed":
        raise ValueError("sampling.mode='fixed' is not supported")
    if mode not in {"independent", "hierarchical"}:
        raise ValueError("sampling.mode must be one of {'independent', 'hierarchical'}")

    space = _coerce_non_empty_str(sampling_cfg.get("space", "param"), field_name="sampling.space")
    if space not in {"param", "z"}:
        raise ValueError("sampling.space must be one of {'param', 'z'}")

    by_condition_raw = sampling_cfg.get("by_condition", {})
    by_condition = _require_mapping(by_condition_raw, field_name="sampling.by_condition")
    if by_condition:
        return _sample_true_parameter_sets_by_condition(
            sampling_cfg=sampling_cfg,
            by_condition=by_condition,
            mode=mode,
            space=space,
            n_parameter_sets=n_parameter_sets,
            seed=seed,
        )

    return _sample_true_parameter_sets_flat(
        sampling_cfg=sampling_cfg,
        mode=mode,
        space=space,
        n_parameter_sets=n_parameter_sets,
        seed=seed,
    )


def _sample_true_parameter_sets_flat(
    *,
    sampling_cfg: dict[str, Any],
    mode: str,
    space: str,
    n_parameter_sets: int,
    seed: int,
) -> tuple[dict[str, float], ...]:
    """Sample parameter sets without per-condition shared+delta remapping."""

    rng = np.random.default_rng(seed)
    clip_to_bounds = bool(sampling_cfg.get("clip_to_bounds", True))
    bounds = _coerce_bounds_mapping(sampling_cfg.get("bounds"), field_name="sampling.bounds")

    if mode == "independent":
        distributions = _require_mapping(
            sampling_cfg.get("distributions"),
            field_name="sampling.distributions",
        )
        if not distributions:
            raise ValueError("sampling.distributions must not be empty")
        parameter_names = tuple(sorted(str(name) for name in distributions))
        transforms = _parse_sampling_transforms(
            sampling_cfg.get("transforms"),
            field_name="sampling.transforms",
            parameter_names=parameter_names,
        )

        draws_by_parameter: dict[str, np.ndarray] = {}
        for parameter_name in parameter_names:
            draws_by_parameter[parameter_name] = _sample_parameter_distribution(
                spec=_require_mapping(
                    distributions.get(parameter_name),
                    field_name=f"sampling.distributions.{parameter_name}",
                ),
                n_draws=n_parameter_sets,
                rng=rng,
                field_name=f"sampling.distributions.{parameter_name}",
            )

        independent_sets: list[dict[str, float]] = []
        for case_index in range(n_parameter_sets):
            sampled_case: dict[str, float] = {}
            for parameter_name in parameter_names:
                raw_value = float(draws_by_parameter[parameter_name][case_index])
                if space == "z":
                    sampled_case[parameter_name] = float(transforms[parameter_name].forward(raw_value))
                else:
                    sampled_case[parameter_name] = raw_value
            if clip_to_bounds and bounds:
                sampled_case = _clip_params_to_bounds(sampled_case, bounds=bounds)
            independent_sets.append(sampled_case)
        return tuple(independent_sets)

    population = _require_mapping(
        sampling_cfg.get("population"),
        field_name="sampling.population",
    )
    individual_sd = _coerce_float_mapping(
        sampling_cfg.get("individual_sd"),
        field_name="sampling.individual_sd",
    )
    if not population:
        raise ValueError("sampling.population must not be empty")
    parameter_names = tuple(sorted(str(name) for name in population))
    missing_sd = [name for name in parameter_names if name not in individual_sd]
    if missing_sd:
        raise ValueError(f"sampling.individual_sd missing parameters: {missing_sd}")
    transforms = _parse_sampling_transforms(
        sampling_cfg.get("transforms"),
        field_name="sampling.transforms",
        parameter_names=parameter_names,
    )

    population_draws: dict[str, float] = {}
    for parameter_name in parameter_names:
        dist_spec = _require_mapping(
            population.get(parameter_name),
            field_name=f"sampling.population.{parameter_name}",
        )
        population_draws[parameter_name] = float(
            _sample_parameter_distribution(
                spec=dist_spec,
                n_draws=1,
                rng=rng,
                field_name=f"sampling.population.{parameter_name}",
            )[0]
        )

    hierarchical_sets: list[dict[str, float]] = []
    for _ in range(n_parameter_sets):
        hierarchical_case: dict[str, float] = {}
        for parameter_name in parameter_names:
            sd = float(individual_sd[parameter_name])
            if sd < 0.0:
                raise ValueError(f"sampling.individual_sd.{parameter_name} must be >= 0")
            case_value = float(rng.normal(loc=population_draws[parameter_name], scale=sd))
            if space == "z":
                hierarchical_case[parameter_name] = float(transforms[parameter_name].forward(case_value))
            else:
                hierarchical_case[parameter_name] = case_value
        if clip_to_bounds and bounds:
            hierarchical_case = _clip_params_to_bounds(hierarchical_case, bounds=bounds)
        hierarchical_sets.append(hierarchical_case)
    return tuple(hierarchical_sets)


def _sample_true_parameter_sets_by_condition(
    *,
    sampling_cfg: dict[str, Any],
    by_condition: dict[str, Any],
    mode: str,
    space: str,
    n_parameter_sets: int,
    seed: int,
) -> tuple[dict[str, float], ...]:
    """Sample and remap condition-wise params into shared+delta ``*_z`` keys."""

    raw_conditions = sampling_cfg.get("conditions")
    if raw_conditions is None:
        conditions = tuple(str(name) for name in by_condition.keys())
    else:
        condition_seq = _require_sequence(raw_conditions, field_name="sampling.conditions")
        conditions = tuple(str(item) for item in condition_seq)
    if not conditions:
        raise ValueError("sampling.by_condition requires at least one condition")

    baseline_condition = _coerce_non_empty_str(
        sampling_cfg.get("baseline_condition"),
        field_name="sampling.baseline_condition",
    )
    if baseline_condition not in conditions:
        raise ValueError("sampling.baseline_condition must be one of sampling.conditions")

    unknown_conditions = sorted(set(by_condition).difference(conditions))
    if unknown_conditions:
        raise ValueError(
            f"sampling.by_condition contains unknown conditions: {unknown_conditions}"
        )

    if mode == "independent":
        base_map = _require_mapping(
            sampling_cfg.get("distributions", {}),
            field_name="sampling.distributions",
        )
        field_name = "distributions"
    else:
        base_map = _require_mapping(
            sampling_cfg.get("population", {}),
            field_name="sampling.population",
        )
        field_name = "population"

    parameter_names = _resolve_condition_parameter_names(
        base_map=base_map,
        by_condition=by_condition,
        field_name=field_name,
    )
    transforms = _parse_sampling_transforms(
        sampling_cfg.get("transforms"),
        field_name="sampling.transforms",
        parameter_names=parameter_names,
    )
    bounds = _coerce_bounds_mapping(sampling_cfg.get("bounds"), field_name="sampling.bounds")
    clip_to_bounds = bool(sampling_cfg.get("clip_to_bounds", True))
    rng = np.random.default_rng(seed)

    if mode == "independent":
        independent_shared_delta_sets: list[dict[str, float]] = []
        for _ in range(n_parameter_sets):
            sampled_z_by_condition: dict[str, dict[str, float]] = {}
            for condition in conditions:
                dist_map = _merge_condition_parameter_map(
                    base_map=base_map,
                    by_condition=by_condition,
                    condition=condition,
                    field_name=field_name,
                )
                _validate_condition_parameter_coverage(
                    mapping=dist_map,
                    parameter_names=parameter_names,
                    field_name=f"sampling.by_condition.{condition}.{field_name}",
                )
                sampled_z_by_condition[condition] = _sample_condition_z_values(
                    distributions=dist_map,
                    parameter_names=parameter_names,
                    transforms=transforms,
                    space=space,
                    clip_to_bounds=clip_to_bounds,
                    bounds=bounds,
                    rng=rng,
                    field_name=f"sampling.by_condition.{condition}.{field_name}",
                )
            independent_shared_delta_sets.append(
                _shared_delta_params_from_condition_z(
                    z_by_condition=sampled_z_by_condition,
                    parameter_names=parameter_names,
                    conditions=conditions,
                    baseline_condition=baseline_condition,
                )
            )
        return tuple(independent_shared_delta_sets)

    base_sd = _coerce_float_mapping(
        sampling_cfg.get("individual_sd", {}),
        field_name="sampling.individual_sd",
    )
    for parameter_name in parameter_names:
        if parameter_name not in base_sd and all(
            parameter_name not in _condition_individual_sd_map(by_condition, condition)
            for condition in conditions
        ):
            raise ValueError(
                "sampling.individual_sd missing parameter "
                f"{parameter_name!r} across all conditions"
            )

    condition_population_z: dict[str, dict[str, float]] = {}
    for condition in conditions:
        pop_map = _merge_condition_parameter_map(
            base_map=base_map,
            by_condition=by_condition,
            condition=condition,
            field_name="population",
        )
        _validate_condition_parameter_coverage(
            mapping=pop_map,
            parameter_names=parameter_names,
            field_name=f"sampling.by_condition.{condition}.population",
        )
        condition_population_z[condition] = _sample_condition_z_values(
            distributions=pop_map,
            parameter_names=parameter_names,
            transforms=transforms,
            space=space,
            clip_to_bounds=clip_to_bounds,
            bounds=bounds,
            rng=rng,
            field_name=f"sampling.by_condition.{condition}.population",
        )

    hierarchical_shared_delta_sets: list[dict[str, float]] = []
    for _ in range(n_parameter_sets):
        hierarchical_z_by_condition: dict[str, dict[str, float]] = {}
        for condition in conditions:
            sd_map = dict(base_sd)
            sd_map.update(_condition_individual_sd_map(by_condition, condition))
            _validate_condition_parameter_coverage(
                mapping=sd_map,
                parameter_names=parameter_names,
                field_name=f"sampling.by_condition.{condition}.individual_sd",
            )
            z_values: dict[str, float] = {}
            for parameter_name in parameter_names:
                sd = float(sd_map[parameter_name])
                if sd < 0.0:
                    raise ValueError(
                        "sampling.by_condition."
                        f"{condition}.individual_sd.{parameter_name} must be >= 0"
                    )
                z_sample = float(
                    rng.normal(
                        loc=condition_population_z[condition][parameter_name],
                        scale=sd,
                    )
                )
                if clip_to_bounds and bounds:
                    theta_sample = float(transforms[parameter_name].forward(z_sample))
                    clipped = _clip_params_to_bounds(
                        {parameter_name: theta_sample},
                        bounds=bounds,
                    )[parameter_name]
                    z_sample = float(transforms[parameter_name].inverse(clipped))
                z_values[parameter_name] = z_sample
            hierarchical_z_by_condition[condition] = z_values
        hierarchical_shared_delta_sets.append(
            _shared_delta_params_from_condition_z(
                z_by_condition=hierarchical_z_by_condition,
                parameter_names=parameter_names,
                conditions=conditions,
                baseline_condition=baseline_condition,
            )
        )
    return tuple(hierarchical_shared_delta_sets)


def _sample_condition_z_values(
    *,
    distributions: Mapping[str, Any],
    parameter_names: tuple[str, ...],
    transforms: Mapping[str, ParameterTransform],
    space: str,
    clip_to_bounds: bool,
    bounds: Mapping[str, tuple[float | None, float | None]] | None,
    rng: np.random.Generator,
    field_name: str,
) -> dict[str, float]:
    """Sample one condition parameter draw and return z-space values."""

    z_values: dict[str, float] = {}
    for parameter_name in parameter_names:
        spec = _require_mapping(
            distributions.get(parameter_name),
            field_name=f"{field_name}.{parameter_name}",
        )
        raw_value = float(
            _sample_parameter_distribution(
                spec=spec,
                n_draws=1,
                rng=rng,
                field_name=f"{field_name}.{parameter_name}",
            )[0]
        )
        if space == "z":
            z_sample = raw_value
            if clip_to_bounds and bounds:
                theta_sample = float(transforms[parameter_name].forward(z_sample))
                clipped = _clip_params_to_bounds(
                    {parameter_name: theta_sample},
                    bounds=bounds,
                )[parameter_name]
                z_sample = float(transforms[parameter_name].inverse(clipped))
            z_values[parameter_name] = z_sample
            continue

        theta_sample = raw_value
        if clip_to_bounds and bounds:
            theta_sample = _clip_params_to_bounds(
                {parameter_name: theta_sample},
                bounds=bounds,
            )[parameter_name]
        z_values[parameter_name] = float(transforms[parameter_name].inverse(theta_sample))
    return z_values


def _shared_delta_params_from_condition_z(
    *,
    z_by_condition: Mapping[str, Mapping[str, float]],
    parameter_names: tuple[str, ...],
    conditions: tuple[str, ...],
    baseline_condition: str,
) -> dict[str, float]:
    """Convert condition-wise z values into shared+delta parameter naming."""

    if baseline_condition not in z_by_condition:
        raise ValueError(
            f"baseline condition {baseline_condition!r} is missing from sampled condition draws"
        )
    baseline = z_by_condition[baseline_condition]
    out: dict[str, float] = {}
    for parameter_name in parameter_names:
        out[f"{parameter_name}__shared_z"] = float(baseline[parameter_name])

    for condition in conditions:
        if condition == baseline_condition:
            continue
        condition_values = z_by_condition[condition]
        for parameter_name in parameter_names:
            out[f"{parameter_name}__delta_z__{condition}"] = float(
                condition_values[parameter_name] - baseline[parameter_name]
            )
    return out


def _resolve_condition_parameter_names(
    *,
    base_map: Mapping[str, Any],
    by_condition: Mapping[str, Any],
    field_name: str,
) -> tuple[str, ...]:
    """Collect parameter names from top-level and by-condition maps."""

    names: set[str] = {str(name) for name in base_map}
    for condition_name, condition_raw in by_condition.items():
        condition_map = _require_mapping(
            condition_raw,
            field_name=f"sampling.by_condition.{condition_name}",
        )
        override_map = _require_mapping(
            condition_map.get(field_name, {}),
            field_name=f"sampling.by_condition.{condition_name}.{field_name}",
        )
        names.update(str(name) for name in override_map)
    if not names:
        raise ValueError(
            "sampling.by_condition requires parameter distributions via "
            f"sampling.{field_name} and/or per-condition overrides"
        )
    return tuple(sorted(names))


def _merge_condition_parameter_map(
    *,
    base_map: Mapping[str, Any],
    by_condition: Mapping[str, Any],
    condition: str,
    field_name: str,
) -> dict[str, Any]:
    """Merge top-level and per-condition parameter map with override semantics."""

    merged = dict(base_map)
    condition_map = _require_mapping(
        by_condition.get(condition, {}),
        field_name=f"sampling.by_condition.{condition}",
    )
    override = _require_mapping(
        condition_map.get(field_name, {}),
        field_name=f"sampling.by_condition.{condition}.{field_name}",
    )
    merged.update(override)
    return merged


def _condition_individual_sd_map(
    by_condition: Mapping[str, Any],
    condition: str,
) -> dict[str, float]:
    """Return per-condition ``individual_sd`` overrides as float mapping."""

    condition_map = _require_mapping(
        by_condition.get(condition, {}),
        field_name=f"sampling.by_condition.{condition}",
    )
    return _coerce_float_mapping(
        condition_map.get("individual_sd", {}),
        field_name=f"sampling.by_condition.{condition}.individual_sd",
    )


def _validate_condition_parameter_coverage(
    *,
    mapping: Mapping[str, Any],
    parameter_names: tuple[str, ...],
    field_name: str,
) -> None:
    """Validate that one condition map covers all required parameter names."""

    missing = [name for name in parameter_names if name not in mapping]
    if missing:
        raise ValueError(f"{field_name} missing parameters: {missing}")


def _parse_sampling_transforms(
    raw: Any,
    *,
    field_name: str,
    parameter_names: tuple[str, ...],
) -> dict[str, ParameterTransform]:
    """Parse sampling transform mapping with default identity transforms."""

    mapping = _require_mapping(raw if raw is not None else {}, field_name=field_name)
    unknown = sorted(set(str(name) for name in mapping).difference(parameter_names))
    if unknown:
        raise ValueError(
            f"{field_name} contains unknown parameters {unknown}; "
            f"expected subset of {list(parameter_names)!r}"
        )

    transforms = {name: identity_transform() for name in parameter_names}
    for param_name, spec in mapping.items():
        if isinstance(spec, str):
            transforms[str(param_name)] = _sampling_transform_from_name(spec)
            continue
        spec_mapping = _require_mapping(
            spec,
            field_name=f"{field_name}.{param_name}",
        )
        kind = _coerce_non_empty_str(
            spec_mapping.get("kind"),
            field_name=f"{field_name}.{param_name}.kind",
        )
        transforms[str(param_name)] = _sampling_transform_from_name(kind)
    return transforms


def _sampling_transform_from_name(name: str) -> ParameterTransform:
    """Resolve one named transform for sampling in z-space."""

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


def _coerce_bounds_mapping(
    raw: Any,
    *,
    field_name: str,
) -> dict[str, tuple[float | None, float | None]] | None:
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


def _clip_params_to_bounds(
    params: Mapping[str, float],
    *,
    bounds: Mapping[str, tuple[float | None, float | None]],
) -> dict[str, float]:
    """Clip parameter values to configured bounds."""

    clipped = {str(name): float(value) for name, value in params.items()}
    for parameter_name, value in list(clipped.items()):
        if parameter_name not in bounds:
            continue
        lower, upper = bounds[parameter_name]
        if lower is not None and value < lower:
            value = float(lower)
        if upper is not None and value > upper:
            value = float(upper)
        clipped[parameter_name] = float(value)
    return clipped


def _sample_parameter_distribution(
    *,
    spec: dict[str, Any],
    n_draws: int,
    rng: np.random.Generator,
    field_name: str,
) -> np.ndarray:
    """Sample one parameter vector from a validated distribution spec."""

    distribution = _coerce_non_empty_str(spec.get("distribution"), field_name=f"{field_name}.distribution")

    if distribution == "uniform":
        validate_allowed_keys(
            spec,
            field_name=field_name,
            allowed_keys=("distribution", "lower", "upper"),
        )
        if "lower" not in spec or "upper" not in spec:
            raise ValueError(f"{field_name} requires 'lower' and 'upper'")
        lower = float(spec["lower"])
        upper = float(spec["upper"])
        if lower > upper:
            raise ValueError(f"{field_name} has lower > upper")
        if lower == upper:
            return np.full(shape=n_draws, fill_value=lower, dtype=float)
        return rng.uniform(lower, upper, size=n_draws)

    if distribution == "normal":
        validate_allowed_keys(
            spec,
            field_name=field_name,
            allowed_keys=("distribution", "mean", "std"),
        )
        if "mean" not in spec or "std" not in spec:
            raise ValueError(f"{field_name} requires 'mean' and 'std'")
        mean = float(spec["mean"])
        std = float(spec["std"])
        if std <= 0.0:
            raise ValueError(f"{field_name}.std must be > 0")
        return rng.normal(loc=mean, scale=std, size=n_draws)

    if distribution == "beta":
        validate_allowed_keys(
            spec,
            field_name=field_name,
            allowed_keys=("distribution", "alpha", "beta"),
        )
        if "alpha" not in spec or "beta" not in spec:
            raise ValueError(f"{field_name} requires 'alpha' and 'beta'")
        alpha = float(spec["alpha"])
        beta = float(spec["beta"])
        if alpha <= 0.0 or beta <= 0.0:
            raise ValueError(f"{field_name}.alpha and {field_name}.beta must be > 0")
        return rng.beta(alpha, beta, size=n_draws)

    if distribution == "log_normal":
        validate_allowed_keys(
            spec,
            field_name=field_name,
            allowed_keys=("distribution", "mean_log", "std_log"),
        )
        if "mean_log" not in spec or "std_log" not in spec:
            raise ValueError(f"{field_name} requires 'mean_log' and 'std_log'")
        mean_log = float(spec["mean_log"])
        std_log = float(spec["std_log"])
        if std_log <= 0.0:
            raise ValueError(f"{field_name}.std_log must be > 0")
        return rng.lognormal(mean=mean_log, sigma=std_log, size=n_draws)

    raise ValueError(
        f"{field_name}.distribution must be one of "
        "{'uniform', 'normal', 'beta', 'log_normal'}"
    )


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


def _parse_selection_criterion(raw: Any, *, field_name: str) -> SelectionCriterion:
    """Parse criterion name into strict model-selection literal type."""

    value = _coerce_non_empty_str(raw, field_name=field_name)
    if value not in {"log_likelihood", "aic", "bic", "waic", "psis_loo"}:
        raise ValueError(
            f"{field_name} must be one of "
            "{'log_likelihood', 'aic', 'bic', 'waic', 'psis_loo'}"
        )
    return cast(SelectionCriterion, value)


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
