"""Config-driven model-comparison fitting helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from comp_model.core.config_validation import validate_allowed_keys
from comp_model.core.data import BlockData, StudyData, SubjectData, TrialDecision
from comp_model.core.events import EpisodeTrace
from comp_model.plugins import PluginRegistry, build_default_registry

from .bayes import build_map_fit_function
from .bayes_config import map_fit_spec_from_config, prior_program_from_config
from .config import fit_spec_from_config, model_component_spec_from_config
from .config_dispatch import MAP_ESTIMATORS, MCMC_ESTIMATORS, MLE_ESTIMATORS
from .fitting import build_model_fit_function
from .likelihood import LikelihoodProgram
from .likelihood_config import likelihood_program_from_config
from .mcmc import sample_posterior_model
from .mcmc_config import mcmc_estimator_spec_from_config
from .model_selection import CandidateFitSpec, ModelComparisonResult, compare_candidate_models
from .study_model_selection import (
    StudyModelComparisonResult,
    SubjectModelComparisonResult,
    compare_study_candidate_models,
    compare_subject_candidate_models,
)


def build_fit_function_from_model_config(
    *,
    model_cfg: Mapping[str, Any],
    estimator_cfg: Mapping[str, Any],
    prior_cfg: Mapping[str, Any] | None,
    registry: PluginRegistry,
    likelihood_cfg: Mapping[str, Any] | None = None,
    likelihood_program: LikelihoodProgram | None = None,
):
    """Build a trace->fit callable from declarative model/estimator config.

    Parameters
    ----------
    model_cfg : Mapping[str, Any]
        Model config with ``component_id`` and optional ``kwargs``.
    estimator_cfg : Mapping[str, Any]
        Estimator config mapping.
    prior_cfg : Mapping[str, Any] | None
        Prior config for MAP estimators. Required for MAP estimator types.
    registry : PluginRegistry
        Plugin registry used to resolve model components.
    likelihood_cfg : Mapping[str, Any] | None, optional
        Optional declarative likelihood config mapping.
    likelihood_program : LikelihoodProgram | None, optional
        Optional likelihood evaluator.

    Returns
    -------
    Callable[[EpisodeTrace], Any]
        Fit function that accepts canonical episode traces.

    Raises
    ------
    ValueError
        If estimator type is unsupported or MAP prior is missing.
    """

    model_spec = model_component_spec_from_config(model_cfg)
    manifest = registry.get("model", model_spec.component_id)
    fixed_kwargs = dict(model_spec.kwargs)
    resolved_likelihood = (
        likelihood_program
        if likelihood_program is not None
        else likelihood_program_from_config(likelihood_cfg)
    )

    model_factory = lambda params: registry.create_model(
        model_spec.component_id,
        **_merge_kwargs(fixed_kwargs, params),
    )

    estimator_type = _coerce_non_empty_str(
        estimator_cfg.get("type"),
        field_name="estimator.type",
    )
    if estimator_type in MLE_ESTIMATORS:
        fit_spec = fit_spec_from_config(estimator_cfg)
        return build_model_fit_function(
            model_factory=model_factory,
            fit_spec=fit_spec,
            requirements=manifest.requirements,
            likelihood_program=resolved_likelihood,
        )

    if estimator_type in MAP_ESTIMATORS:
        if prior_cfg is None:
            raise ValueError(
                f"prior is required for estimator type {estimator_type!r}"
            )
        prior_program = prior_program_from_config(prior_cfg)
        fit_spec = map_fit_spec_from_config(estimator_cfg)
        return build_map_fit_function(
            model_factory=model_factory,
            prior_program=prior_program,
            fit_spec=fit_spec,
            requirements=manifest.requirements,
            likelihood_program=resolved_likelihood,
        )

    if estimator_type in MCMC_ESTIMATORS:
        if prior_cfg is None:
            raise ValueError(
                f"prior is required for estimator type {estimator_type!r}"
            )
        prior_program = prior_program_from_config(prior_cfg)
        estimator_spec = mcmc_estimator_spec_from_config(estimator_cfg)
        return lambda trace: sample_posterior_model(
            trace,
            model_factory=model_factory,
            prior_program=prior_program,
            initial_params=estimator_spec.initial_params,
            n_samples=estimator_spec.n_samples,
            n_warmup=estimator_spec.n_warmup,
            thin=estimator_spec.thin,
            proposal_scales=estimator_spec.proposal_scales,
            bounds=estimator_spec.bounds,
            requirements=manifest.requirements,
            likelihood_program=resolved_likelihood,
            random_seed=estimator_spec.random_seed,
        )

    supported = sorted(MLE_ESTIMATORS | MAP_ESTIMATORS | MCMC_ESTIMATORS)
    raise ValueError(
        f"estimator.type must be one of {supported}; got {estimator_type!r}"
    )


def compare_dataset_candidates_from_config(
    data: EpisodeTrace | BlockData | tuple[TrialDecision, ...] | list[TrialDecision],
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> ModelComparisonResult:
    """Compare configured model candidates on one dataset.

    Parameters
    ----------
    data : EpisodeTrace | BlockData | tuple[TrialDecision, ...] | list[TrialDecision]
        Dataset container.
    config : Mapping[str, Any]
        Config with keys:
        ``candidates`` (list of candidate entries), optional ``criterion``,
        optional ``n_observations``.
    registry : PluginRegistry | None, optional
        Optional plugin registry.
    likelihood_program : LikelihoodProgram | None, optional
        Optional likelihood evaluator.

    Returns
    -------
    ModelComparisonResult
        Candidate-model comparison output.
    """

    cfg = _require_mapping(config, field_name="config")
    validate_allowed_keys(
        cfg,
        field_name="config",
        allowed_keys=("candidates", "criterion", "n_observations", "likelihood"),
    )
    criterion = str(cfg.get("criterion", "log_likelihood"))
    n_observations = int(cfg["n_observations"]) if "n_observations" in cfg else None
    likelihood_cfg = (
        _require_mapping(cfg.get("likelihood"), field_name="config.likelihood")
        if "likelihood" in cfg
        else None
    )

    candidate_specs = _candidate_specs_from_config(
        cfg=cfg,
        registry=registry,
        likelihood_cfg=likelihood_cfg,
        likelihood_program=likelihood_program,
    )

    return compare_candidate_models(
        data,
        candidate_specs=candidate_specs,
        criterion=criterion,
        n_observations=n_observations,
    )


def compare_subject_candidates_from_config(
    subject: SubjectData,
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> SubjectModelComparisonResult:
    """Compare configured model candidates across all blocks for one subject."""

    cfg = _require_mapping(config, field_name="config")
    validate_allowed_keys(
        cfg,
        field_name="config",
        allowed_keys=("candidates", "criterion", "likelihood"),
    )
    likelihood_cfg = (
        _require_mapping(cfg.get("likelihood"), field_name="config.likelihood")
        if "likelihood" in cfg
        else None
    )
    candidate_specs = _candidate_specs_from_config(
        cfg=cfg,
        registry=registry,
        likelihood_cfg=likelihood_cfg,
        likelihood_program=likelihood_program,
    )
    criterion = str(cfg.get("criterion", "log_likelihood"))
    return compare_subject_candidate_models(
        subject,
        candidate_specs=candidate_specs,
        criterion=criterion,
    )


def compare_study_candidates_from_config(
    study: StudyData,
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> StudyModelComparisonResult:
    """Compare configured model candidates across all study subjects."""

    cfg = _require_mapping(config, field_name="config")
    validate_allowed_keys(
        cfg,
        field_name="config",
        allowed_keys=("candidates", "criterion", "likelihood"),
    )
    likelihood_cfg = (
        _require_mapping(cfg.get("likelihood"), field_name="config.likelihood")
        if "likelihood" in cfg
        else None
    )
    candidate_specs = _candidate_specs_from_config(
        cfg=cfg,
        registry=registry,
        likelihood_cfg=likelihood_cfg,
        likelihood_program=likelihood_program,
    )
    criterion = str(cfg.get("criterion", "log_likelihood"))
    return compare_study_candidate_models(
        study,
        candidate_specs=candidate_specs,
        criterion=criterion,
    )


def _candidate_specs_from_config(
    *,
    cfg: Mapping[str, Any],
    registry: PluginRegistry | None,
    likelihood_cfg: Mapping[str, Any] | None,
    likelihood_program: LikelihoodProgram | None,
) -> tuple[CandidateFitSpec, ...]:
    """Parse and build candidate fit specs from shared config format."""

    candidate_rows = _require_sequence(cfg.get("candidates"), field_name="config.candidates")
    reg = registry if registry is not None else build_default_registry()
    candidate_specs: list[CandidateFitSpec] = []
    for index, raw in enumerate(candidate_rows):
        item = _require_mapping(raw, field_name=f"config.candidates[{index}]")
        validate_allowed_keys(
            item,
            field_name=f"config.candidates[{index}]",
            allowed_keys=("name", "model", "estimator", "prior", "likelihood", "n_parameters"),
        )
        name = _coerce_non_empty_str(item.get("name"), field_name=f"config.candidates[{index}].name")
        model_cfg = _require_mapping(item.get("model"), field_name=f"config.candidates[{index}].model")
        estimator_cfg = _require_mapping(item.get("estimator"), field_name=f"config.candidates[{index}].estimator")
        prior_cfg = (
            _require_mapping(item["prior"], field_name=f"config.candidates[{index}].prior")
            if "prior" in item
            else None
        )
        candidate_likelihood_cfg = (
            _require_mapping(
                item.get("likelihood"),
                field_name=f"config.candidates[{index}].likelihood",
            )
            if "likelihood" in item
            else likelihood_cfg
        )

        fit_function = build_fit_function_from_model_config(
            model_cfg=model_cfg,
            estimator_cfg=estimator_cfg,
            prior_cfg=prior_cfg,
            registry=reg,
            likelihood_cfg=candidate_likelihood_cfg,
            likelihood_program=likelihood_program,
        )

        n_parameters_raw = item.get("n_parameters")
        n_parameters = int(n_parameters_raw) if n_parameters_raw is not None else None
        candidate_specs.append(
            CandidateFitSpec(
                name=name,
                fit_function=fit_function,
                n_parameters=n_parameters,
            )
        )

    return tuple(candidate_specs)


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


def _merge_kwargs(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge fixed keyword arguments with free-parameter overrides."""

    merged = dict(base)
    merged.update(dict(override))
    return merged


__all__ = [
    "build_fit_function_from_model_config",
    "compare_dataset_candidates_from_config",
    "compare_study_candidates_from_config",
    "compare_subject_candidates_from_config",
]
