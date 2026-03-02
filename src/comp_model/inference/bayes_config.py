"""Config-driven Bayesian fitting helpers.

This module mirrors the MLE config entry points for MAP and within-subject
hierarchical MAP fitting, using declarative mapping/JSON-style configurations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from comp_model.core.config_validation import validate_allowed_keys
from comp_model.core.data import BlockData, StudyData, SubjectData, TrialDecision
from comp_model.core.events import EpisodeTrace
from comp_model.plugins import PluginRegistry, build_default_registry

from .bayes import (
    BayesFitResult,
    IndependentPriorProgram,
    MapEstimatorType,
    MapFitSpec,
    PriorProgram,
    beta_log_prior,
    fit_map_model_from_registry,
    log_normal_log_prior,
    normal_log_prior,
    uniform_log_prior,
)
from .block_strategy import BlockFitStrategy, coerce_block_fit_strategy
from .config import model_component_spec_from_config
from .hierarchical import (
    HierarchicalStudyMapResult,
    HierarchicalSubjectMapResult,
    fit_study_hierarchical_map,
    fit_subject_hierarchical_map,
)
from .likelihood import LikelihoodProgram
from .likelihood_config import likelihood_program_from_config
from .map_study_fitting import (
    MapBlockFitResult,
    MapStudyFitResult,
    MapSubjectFitResult,
    fit_map_block_data,
    fit_map_study_data,
    fit_map_subject_data,
)
from .transforms import (
    ParameterTransform,
    identity_transform,
    positive_log_transform,
    unit_interval_logit_transform,
)


@dataclass(frozen=True, slots=True)
class HierarchicalMapEstimatorSpec:
    """Parsed estimator spec for within-subject hierarchical MAP.

    Parameters
    ----------
    parameter_names : tuple[str, ...]
        Hierarchically pooled parameter names.
    transforms : dict[str, ParameterTransform] | None
        Optional per-parameter transforms.
    initial_group_location : dict[str, float] | None
        Optional initial constrained group-location values.
    initial_group_scale : dict[str, float] | None
        Optional initial positive group-scale values.
    initial_block_params : tuple[dict[str, float], ...] | None
        Optional shared initial block parameters used in subject fitting.
    initial_block_params_by_subject : dict[str, tuple[dict[str, float], ...]] | None
        Optional subject-specific initial block parameters used in study fitting.
    mu_prior_mean : float
        Group-location Normal prior mean in unconstrained space.
    mu_prior_std : float
        Group-location Normal prior standard deviation.
    log_sigma_prior_mean : float
        Prior mean for log group scale.
    log_sigma_prior_std : float
        Prior standard deviation for log group scale.
    method : str
        SciPy minimizer method.
    tol : float | None
        SciPy minimizer tolerance.
    """

    parameter_names: tuple[str, ...]
    transforms: dict[str, ParameterTransform] | None = None
    initial_group_location: dict[str, float] | None = None
    initial_group_scale: dict[str, float] | None = None
    initial_block_params: tuple[dict[str, float], ...] | None = None
    initial_block_params_by_subject: dict[str, tuple[dict[str, float], ...]] | None = None
    mu_prior_mean: float = 0.0
    mu_prior_std: float = 2.0
    log_sigma_prior_mean: float = -1.0
    log_sigma_prior_std: float = 1.0
    method: str = "L-BFGS-B"
    tol: float | None = None


def map_fit_spec_from_config(estimator_cfg: Mapping[str, Any]) -> MapFitSpec:
    """Parse MAP estimator config into :class:`MapFitSpec`.

    Parameters
    ----------
    estimator_cfg : Mapping[str, Any]
        Estimator configuration mapping.

    Returns
    -------
    MapFitSpec
        Parsed MAP fit specification.
    """

    estimator = _require_mapping(estimator_cfg, field_name="estimator")
    estimator_type_raw = _coerce_non_empty_str(estimator.get("type"), field_name="estimator.type")
    if estimator_type_raw == "scipy_map":
        estimator_type: MapEstimatorType = "scipy_map"
        validate_allowed_keys(
            estimator,
            field_name="estimator",
            allowed_keys=("type", "initial_params", "bounds", "method", "tol"),
        )
    elif estimator_type_raw == "transformed_scipy_map":
        estimator_type = "transformed_scipy_map"
        validate_allowed_keys(
            estimator,
            field_name="estimator",
            allowed_keys=("type", "initial_params", "bounds_z", "transforms", "method", "tol"),
        )
    else:
        raise ValueError(
            "estimator.type must be one of {'scipy_map', 'transformed_scipy_map'}"
        )

    return MapFitSpec(
        estimator_type=estimator_type,
        initial_params=_coerce_float_mapping(
            estimator.get("initial_params"),
            field_name="estimator.initial_params",
        ),
        bounds=(
            _coerce_bounds_mapping(estimator.get("bounds"), field_name="estimator.bounds")
            if estimator_type == "scipy_map"
            else None
        ),
        bounds_z=(
            _coerce_bounds_mapping(estimator.get("bounds_z"), field_name="estimator.bounds_z")
            if estimator_type == "transformed_scipy_map"
            else None
        ),
        transforms=(
            _parse_transforms_mapping(estimator.get("transforms", {}), field_name="estimator.transforms")
            if estimator_type == "transformed_scipy_map"
            else None
        ),
        method=str(estimator.get("method", "L-BFGS-B")),
        tol=float(estimator["tol"]) if "tol" in estimator else None,
    )


def prior_program_from_config(prior_cfg: Mapping[str, Any]) -> PriorProgram:
    """Parse prior configuration mapping into a :class:`PriorProgram`.

    Parameters
    ----------
    prior_cfg : Mapping[str, Any]
        Prior configuration mapping.

    Returns
    -------
    PriorProgram
        Parsed prior program.
    """

    prior = _require_mapping(prior_cfg, field_name="prior")
    prior_type = str(prior.get("type", "independent"))
    if prior_type != "independent":
        raise ValueError("prior.type must be 'independent'")

    parameters_raw = prior.get("parameters")
    if parameters_raw is not None:
        validate_allowed_keys(
            prior,
            field_name="prior",
            allowed_keys=("type", "parameters", "require_all"),
        )
    if parameters_raw is None:
        parameters = {
            key: value
            for key, value in prior.items()
            if key not in {"type", "require_all"}
        }
    else:
        parameters = _require_mapping(parameters_raw, field_name="prior.parameters")

    if not parameters:
        raise ValueError("prior.parameters must include at least one parameter prior")

    log_pdf_by_param = {
        str(name): _prior_log_pdf_from_config(raw, field_name=f"prior.parameters.{name}")
        for name, raw in parameters.items()
    }
    require_all = bool(prior.get("require_all", True))
    return IndependentPriorProgram(log_pdf_by_param=log_pdf_by_param, require_all=require_all)


def hierarchical_map_spec_from_config(estimator_cfg: Mapping[str, Any]) -> HierarchicalMapEstimatorSpec:
    """Parse within-subject hierarchical MAP estimator config.

    Parameters
    ----------
    estimator_cfg : Mapping[str, Any]
        Estimator configuration mapping.

    Returns
    -------
    HierarchicalMapEstimatorSpec
        Parsed hierarchical estimator specification.
    """

    estimator = _require_mapping(estimator_cfg, field_name="estimator")
    estimator_type = _coerce_non_empty_str(estimator.get("type"), field_name="estimator.type")
    if estimator_type != "within_subject_hierarchical_map":
        raise ValueError(
            "estimator.type must be 'within_subject_hierarchical_map' "
            "for hierarchical MAP config"
        )
    validate_allowed_keys(
        estimator,
        field_name="estimator",
        allowed_keys=(
            "type",
            "parameter_names",
            "transforms",
            "initial_group_location",
            "initial_group_scale",
            "initial_block_params",
            "initial_block_params_by_subject",
            "mu_prior_mean",
            "mu_prior_std",
            "log_sigma_prior_mean",
            "log_sigma_prior_std",
            "method",
            "tol",
        ),
    )

    raw_names = _require_sequence(estimator.get("parameter_names"), field_name="estimator.parameter_names")
    parameter_names = tuple(
        _coerce_non_empty_str(name, field_name=f"estimator.parameter_names[{index}]")
        for index, name in enumerate(raw_names)
    )
    if len(set(parameter_names)) != len(parameter_names):
        raise ValueError("estimator.parameter_names must be unique")

    initial_block_params = None
    if "initial_block_params" in estimator:
        rows = _require_sequence(estimator["initial_block_params"], field_name="estimator.initial_block_params")
        initial_block_params = tuple(
            _coerce_float_mapping(row, field_name=f"estimator.initial_block_params[{index}]")
            for index, row in enumerate(rows)
        )

    initial_block_params_by_subject = None
    if "initial_block_params_by_subject" in estimator:
        raw_subject_map = _require_mapping(
            estimator["initial_block_params_by_subject"],
            field_name="estimator.initial_block_params_by_subject",
        )
        parsed_subject_map: dict[str, tuple[dict[str, float], ...]] = {}
        for subject_id, raw_sequence in raw_subject_map.items():
            rows = _require_sequence(
                raw_sequence,
                field_name=f"estimator.initial_block_params_by_subject.{subject_id}",
            )
            parsed_subject_map[str(subject_id)] = tuple(
                _coerce_float_mapping(
                    row,
                    field_name=f"estimator.initial_block_params_by_subject.{subject_id}[{index}]",
                )
                for index, row in enumerate(rows)
            )
        initial_block_params_by_subject = parsed_subject_map

    return HierarchicalMapEstimatorSpec(
        parameter_names=parameter_names,
        transforms=(
            _parse_transforms_mapping(estimator.get("transforms", {}), field_name="estimator.transforms")
            if "transforms" in estimator
            else None
        ),
        initial_group_location=(
            _coerce_float_mapping(estimator["initial_group_location"], field_name="estimator.initial_group_location")
            if "initial_group_location" in estimator
            else None
        ),
        initial_group_scale=(
            _coerce_float_mapping(estimator["initial_group_scale"], field_name="estimator.initial_group_scale")
            if "initial_group_scale" in estimator
            else None
        ),
        initial_block_params=initial_block_params,
        initial_block_params_by_subject=initial_block_params_by_subject,
        mu_prior_mean=float(estimator.get("mu_prior_mean", 0.0)),
        mu_prior_std=float(estimator.get("mu_prior_std", 2.0)),
        log_sigma_prior_mean=float(estimator.get("log_sigma_prior_mean", -1.0)),
        log_sigma_prior_std=float(estimator.get("log_sigma_prior_std", 1.0)),
        method=str(estimator.get("method", "L-BFGS-B")),
        tol=float(estimator["tol"]) if "tol" in estimator else None,
    )


def fit_map_dataset_from_config(
    data: EpisodeTrace | BlockData | tuple[TrialDecision, ...] | list[TrialDecision],
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> BayesFitResult:
    """Fit one dataset with MAP using declarative config.

    Parameters
    ----------
    data : EpisodeTrace | BlockData | tuple[TrialDecision, ...] | list[TrialDecision]
        Dataset container.
    config : Mapping[str, Any]
        Config with ``model``, ``prior``, and ``estimator`` sections.
    registry : PluginRegistry | None, optional
        Optional plugin registry.
    likelihood_program : LikelihoodProgram | None, optional
        Optional likelihood evaluator.

    Returns
    -------
    BayesFitResult
        MAP fit output.
    """

    cfg = _require_mapping(config, field_name="config")
    validate_allowed_keys(
        cfg,
        field_name="config",
        allowed_keys=("model", "prior", "estimator", "likelihood"),
    )
    model_spec = model_component_spec_from_config(_require_mapping(cfg.get("model"), field_name="config.model"))
    prior_program = prior_program_from_config(_require_mapping(cfg.get("prior"), field_name="config.prior"))
    fit_spec = map_fit_spec_from_config(_require_mapping(cfg.get("estimator"), field_name="config.estimator"))
    likelihood_cfg = (
        _require_mapping(cfg.get("likelihood"), field_name="config.likelihood")
        if "likelihood" in cfg
        else None
    )
    resolved_likelihood = (
        likelihood_program
        if likelihood_program is not None
        else likelihood_program_from_config(likelihood_cfg)
    )

    reg = registry if registry is not None else build_default_registry()
    return fit_map_model_from_registry(
        data,
        model_component_id=model_spec.component_id,
        prior_program=prior_program,
        fit_spec=fit_spec,
        model_kwargs=model_spec.kwargs,
        registry=reg,
        likelihood_program=resolved_likelihood,
    )


def fit_map_block_from_config(
    block: BlockData,
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> MapBlockFitResult:
    """Fit one block with MAP using declarative config."""

    cfg = _require_mapping(config, field_name="config")
    validate_allowed_keys(
        cfg,
        field_name="config",
        allowed_keys=("model", "prior", "estimator", "likelihood"),
    )
    model_spec = model_component_spec_from_config(_require_mapping(cfg.get("model"), field_name="config.model"))
    prior_program = prior_program_from_config(_require_mapping(cfg.get("prior"), field_name="config.prior"))
    fit_spec = map_fit_spec_from_config(_require_mapping(cfg.get("estimator"), field_name="config.estimator"))
    likelihood_cfg = (
        _require_mapping(cfg.get("likelihood"), field_name="config.likelihood")
        if "likelihood" in cfg
        else None
    )
    resolved_likelihood = (
        likelihood_program
        if likelihood_program is not None
        else likelihood_program_from_config(likelihood_cfg)
    )

    reg = registry if registry is not None else build_default_registry()
    return fit_map_block_data(
        block,
        model_component_id=model_spec.component_id,
        prior_program=prior_program,
        fit_spec=fit_spec,
        model_kwargs=model_spec.kwargs,
        registry=reg,
        likelihood_program=resolved_likelihood,
    )


def fit_map_subject_from_config(
    subject: SubjectData,
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> MapSubjectFitResult:
    """Fit one subject with MAP using declarative config."""

    cfg = _require_mapping(config, field_name="config")
    validate_allowed_keys(
        cfg,
        field_name="config",
        allowed_keys=("model", "prior", "estimator", "likelihood", "block_fit_strategy"),
    )
    model_spec = model_component_spec_from_config(_require_mapping(cfg.get("model"), field_name="config.model"))
    prior_program = prior_program_from_config(_require_mapping(cfg.get("prior"), field_name="config.prior"))
    fit_spec = map_fit_spec_from_config(_require_mapping(cfg.get("estimator"), field_name="config.estimator"))
    likelihood_cfg = (
        _require_mapping(cfg.get("likelihood"), field_name="config.likelihood")
        if "likelihood" in cfg
        else None
    )
    resolved_likelihood = (
        likelihood_program
        if likelihood_program is not None
        else likelihood_program_from_config(likelihood_cfg)
    )
    block_fit_strategy: BlockFitStrategy = coerce_block_fit_strategy(
        cfg.get("block_fit_strategy"),
        field_name="config.block_fit_strategy",
    )

    reg = registry if registry is not None else build_default_registry()
    return fit_map_subject_data(
        subject,
        model_component_id=model_spec.component_id,
        prior_program=prior_program,
        fit_spec=fit_spec,
        model_kwargs=model_spec.kwargs,
        registry=reg,
        likelihood_program=resolved_likelihood,
        block_fit_strategy=block_fit_strategy,
    )


def fit_map_study_from_config(
    study: StudyData,
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> MapStudyFitResult:
    """Fit one study with MAP using declarative config."""

    cfg = _require_mapping(config, field_name="config")
    validate_allowed_keys(
        cfg,
        field_name="config",
        allowed_keys=("model", "prior", "estimator", "likelihood", "block_fit_strategy"),
    )
    model_spec = model_component_spec_from_config(_require_mapping(cfg.get("model"), field_name="config.model"))
    prior_program = prior_program_from_config(_require_mapping(cfg.get("prior"), field_name="config.prior"))
    fit_spec = map_fit_spec_from_config(_require_mapping(cfg.get("estimator"), field_name="config.estimator"))
    likelihood_cfg = (
        _require_mapping(cfg.get("likelihood"), field_name="config.likelihood")
        if "likelihood" in cfg
        else None
    )
    resolved_likelihood = (
        likelihood_program
        if likelihood_program is not None
        else likelihood_program_from_config(likelihood_cfg)
    )
    block_fit_strategy: BlockFitStrategy = coerce_block_fit_strategy(
        cfg.get("block_fit_strategy"),
        field_name="config.block_fit_strategy",
    )

    reg = registry if registry is not None else build_default_registry()
    return fit_map_study_data(
        study,
        model_component_id=model_spec.component_id,
        prior_program=prior_program,
        fit_spec=fit_spec,
        model_kwargs=model_spec.kwargs,
        registry=reg,
        likelihood_program=resolved_likelihood,
        block_fit_strategy=block_fit_strategy,
    )


def fit_subject_hierarchical_map_from_config(
    subject: SubjectData,
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> HierarchicalSubjectMapResult:
    """Fit one subject with within-subject hierarchical MAP from config.

    Parameters
    ----------
    subject : SubjectData
        Subject dataset.
    config : Mapping[str, Any]
        Config with ``model`` and ``estimator`` sections. The estimator type
        must be ``within_subject_hierarchical_map``.
    registry : PluginRegistry | None, optional
        Optional plugin registry.
    likelihood_program : LikelihoodProgram | None, optional
        Optional likelihood evaluator.

    Returns
    -------
    HierarchicalSubjectMapResult
        Subject-level hierarchical MAP output.
    """

    cfg = _require_mapping(config, field_name="config")
    validate_allowed_keys(
        cfg,
        field_name="config",
        allowed_keys=("model", "estimator", "likelihood"),
    )
    model_spec = model_component_spec_from_config(_require_mapping(cfg.get("model"), field_name="config.model"))
    estimator_spec = hierarchical_map_spec_from_config(
        _require_mapping(cfg.get("estimator"), field_name="config.estimator")
    )
    likelihood_cfg = (
        _require_mapping(cfg.get("likelihood"), field_name="config.likelihood")
        if "likelihood" in cfg
        else None
    )
    resolved_likelihood = (
        likelihood_program
        if likelihood_program is not None
        else likelihood_program_from_config(likelihood_cfg)
    )

    reg = registry if registry is not None else build_default_registry()
    manifest = reg.get("model", model_spec.component_id)

    fixed_kwargs = dict(model_spec.kwargs)
    model_factory = lambda params: reg.create_model(
        model_spec.component_id,
        **_merge_kwargs(fixed_kwargs, params),
    )

    subject_block_init = estimator_spec.initial_block_params
    if estimator_spec.initial_block_params_by_subject is not None:
        subject_specific = estimator_spec.initial_block_params_by_subject.get(subject.subject_id)
        if subject_specific is not None:
            subject_block_init = subject_specific

    return fit_subject_hierarchical_map(
        subject,
        model_factory=model_factory,
        parameter_names=estimator_spec.parameter_names,
        transforms=estimator_spec.transforms,
        likelihood_program=resolved_likelihood,
        requirements=manifest.requirements,
        initial_group_location=estimator_spec.initial_group_location,
        initial_group_scale=estimator_spec.initial_group_scale,
        initial_block_params=subject_block_init,
        mu_prior_mean=estimator_spec.mu_prior_mean,
        mu_prior_std=estimator_spec.mu_prior_std,
        log_sigma_prior_mean=estimator_spec.log_sigma_prior_mean,
        log_sigma_prior_std=estimator_spec.log_sigma_prior_std,
        method=estimator_spec.method,
        tol=estimator_spec.tol,
    )


def fit_study_hierarchical_map_from_config(
    study: StudyData,
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> HierarchicalStudyMapResult:
    """Fit all study subjects with within-subject hierarchical MAP from config.

    Parameters
    ----------
    study : StudyData
        Study dataset.
    config : Mapping[str, Any]
        Config with ``model`` and ``estimator`` sections. The estimator type
        must be ``within_subject_hierarchical_map``.
    registry : PluginRegistry | None, optional
        Optional plugin registry.
    likelihood_program : LikelihoodProgram | None, optional
        Optional likelihood evaluator.

    Returns
    -------
    HierarchicalStudyMapResult
        Study-level hierarchical MAP output.
    """

    cfg = _require_mapping(config, field_name="config")
    validate_allowed_keys(
        cfg,
        field_name="config",
        allowed_keys=("model", "estimator", "likelihood"),
    )
    model_spec = model_component_spec_from_config(_require_mapping(cfg.get("model"), field_name="config.model"))
    estimator_spec = hierarchical_map_spec_from_config(
        _require_mapping(cfg.get("estimator"), field_name="config.estimator")
    )
    likelihood_cfg = (
        _require_mapping(cfg.get("likelihood"), field_name="config.likelihood")
        if "likelihood" in cfg
        else None
    )
    resolved_likelihood = (
        likelihood_program
        if likelihood_program is not None
        else likelihood_program_from_config(likelihood_cfg)
    )

    reg = registry if registry is not None else build_default_registry()
    manifest = reg.get("model", model_spec.component_id)
    fixed_kwargs = dict(model_spec.kwargs)

    model_factory = lambda params: reg.create_model(
        model_spec.component_id,
        **_merge_kwargs(fixed_kwargs, params),
    )

    return fit_study_hierarchical_map(
        study,
        model_factory=model_factory,
        parameter_names=estimator_spec.parameter_names,
        transforms=estimator_spec.transforms,
        likelihood_program=resolved_likelihood,
        requirements=manifest.requirements,
        initial_group_location=estimator_spec.initial_group_location,
        initial_group_scale=estimator_spec.initial_group_scale,
        initial_block_params_by_subject=estimator_spec.initial_block_params_by_subject,
        mu_prior_mean=estimator_spec.mu_prior_mean,
        mu_prior_std=estimator_spec.mu_prior_std,
        log_sigma_prior_mean=estimator_spec.log_sigma_prior_mean,
        log_sigma_prior_std=estimator_spec.log_sigma_prior_std,
        method=estimator_spec.method,
        tol=estimator_spec.tol,
    )


def _parse_transforms_mapping(raw: Any, *, field_name: str) -> dict[str, ParameterTransform]:
    """Parse configured transform mapping."""

    mapping = _require_mapping(raw, field_name=field_name)
    out: dict[str, ParameterTransform] = {}
    for param_name, spec in mapping.items():
        if isinstance(spec, str):
            out[str(param_name)] = _transform_from_name(spec)
            continue

        spec_mapping = _require_mapping(spec, field_name=f"{field_name}.{param_name}")
        validate_allowed_keys(
            spec_mapping,
            field_name=f"{field_name}.{param_name}",
            allowed_keys=("kind",),
        )
        kind = _coerce_non_empty_str(spec_mapping.get("kind"), field_name=f"{field_name}.{param_name}.kind")
        out[str(param_name)] = _transform_from_name(kind)
    return out


def _transform_from_name(name: str) -> ParameterTransform:
    """Resolve transform name into concrete transform object."""

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


def _prior_log_pdf_from_config(raw: Any, *, field_name: str):
    """Parse one parameter prior specification into a log-density callable."""

    spec = _require_mapping(raw, field_name=field_name)
    distribution = _coerce_non_empty_str(
        spec.get("distribution", spec.get("kind")),
        field_name=f"{field_name}.distribution",
    )

    if distribution == "normal":
        validate_allowed_keys(
            spec,
            field_name=field_name,
            allowed_keys=("distribution", "kind", "mean", "std"),
        )
        return normal_log_prior(
            mean=float(spec.get("mean", 0.0)),
            std=float(spec["std"]),
        )
    if distribution == "uniform":
        validate_allowed_keys(
            spec,
            field_name=field_name,
            allowed_keys=("distribution", "kind", "lower", "upper"),
        )
        lower = float(spec["lower"]) if "lower" in spec else None
        upper = float(spec["upper"]) if "upper" in spec else None
        return uniform_log_prior(lower=lower, upper=upper)
    if distribution == "beta":
        validate_allowed_keys(
            spec,
            field_name=field_name,
            allowed_keys=("distribution", "kind", "alpha", "beta"),
        )
        return beta_log_prior(
            alpha=float(spec["alpha"]),
            beta=float(spec["beta"]),
        )
    if distribution in {"log_normal", "lognormal"}:
        validate_allowed_keys(
            spec,
            field_name=field_name,
            allowed_keys=("distribution", "kind", "mean_log", "std_log"),
        )
        return log_normal_log_prior(
            mean_log=float(spec.get("mean_log", 0.0)),
            std_log=float(spec["std_log"]),
        )
    raise ValueError(
        f"unsupported prior distribution {distribution!r}; expected one of "
        "{'normal', 'uniform', 'beta', 'log_normal'}"
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


def _merge_kwargs(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge fixed keyword arguments with free-parameter overrides."""

    merged = dict(base)
    merged.update(dict(override))
    return merged


__all__ = [
    "HierarchicalMapEstimatorSpec",
    "MapBlockFitResult",
    "MapStudyFitResult",
    "MapSubjectFitResult",
    "fit_map_block_from_config",
    "fit_map_dataset_from_config",
    "fit_map_study_from_config",
    "fit_map_subject_from_config",
    "fit_study_hierarchical_map_from_config",
    "fit_subject_hierarchical_map_from_config",
    "hierarchical_map_spec_from_config",
    "map_fit_spec_from_config",
    "prior_program_from_config",
]
