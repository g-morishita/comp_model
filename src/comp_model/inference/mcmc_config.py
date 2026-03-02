"""Config-driven MCMC posterior sampling helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from comp_model.core.config_validation import validate_allowed_keys
from comp_model.core.data import BlockData, StudyData, SubjectData, TrialDecision
from comp_model.core.events import EpisodeTrace
from comp_model.plugins import PluginRegistry, build_default_registry

from .bayes import PriorProgram
from .bayes_config import prior_program_from_config
from .config import model_component_spec_from_config
from .hierarchical_mcmc import (
    HierarchicalStudyPosteriorResult,
    HierarchicalSubjectPosteriorResult,
    sample_study_hierarchical_posterior,
    sample_subject_hierarchical_posterior,
)
from .hierarchical_stan import (
    sample_study_hierarchical_posterior_stan,
    sample_subject_hierarchical_posterior_stan,
)
from .likelihood import LikelihoodProgram
from .likelihood_config import likelihood_program_from_config
from .mcmc import MCMCPosteriorResult, sample_posterior_model_from_registry
from .mcmc_study_fitting import (
    MCMCBlockResult,
    MCMCStudyResult,
    MCMCSubjectResult,
    sample_posterior_block_data,
    sample_posterior_study_data,
    sample_posterior_subject_data,
)
from .transforms import (
    ParameterTransform,
    identity_transform,
    positive_log_transform,
    unit_interval_logit_transform,
)


@dataclass(frozen=True, slots=True)
class MCMCEstimatorSpec:
    """Parsed estimator spec for random-walk Metropolis sampling.

    Parameters
    ----------
    initial_params : dict[str, float]
        Initial constrained parameter values.
    n_samples : int
        Number of retained posterior draws.
    n_warmup : int
        Number of warmup iterations.
    thin : int
        Thinning interval.
    proposal_scales : dict[str, float] | None
        Optional per-parameter proposal scales.
    bounds : dict[str, tuple[float | None, float | None]] | None
        Optional hard bounds by parameter name.
    random_seed : int | None
        Optional RNG seed.
    """

    initial_params: dict[str, float]
    n_samples: int
    n_warmup: int = 500
    thin: int = 1
    proposal_scales: dict[str, float] | None = None
    bounds: dict[str, tuple[float | None, float | None]] | None = None
    random_seed: int | None = None


@dataclass(frozen=True, slots=True)
class HierarchicalMCMCEstimatorSpec:
    """Parsed estimator spec for within-subject hierarchical MCMC.

    Parameters
    ----------
    parameter_names : tuple[str, ...]
        Names of pooled parameters across blocks.
    transforms : dict[str, ParameterTransform] | None
        Optional per-parameter transforms from unconstrained ``z`` space.
    initial_group_location : dict[str, float] | None
        Optional initial constrained group-location values.
    initial_group_scale : dict[str, float] | None
        Optional initial positive group-scale values.
    initial_block_params : tuple[dict[str, float], ...] | None
        Optional per-block initial constrained parameter values.
    initial_block_params_by_subject : dict[str, tuple[dict[str, float], ...]] | None
        Optional subject-specific per-block initial parameter mappings.
    mu_prior_mean : float
        Group-location prior mean in ``z`` space.
    mu_prior_std : float
        Group-location prior std in ``z`` space.
    log_sigma_prior_mean : float
        Group log-scale prior mean.
    log_sigma_prior_std : float
        Group log-scale prior std.
    n_samples : int
        Number of retained draws after warmup/thinning.
    n_warmup : int
        Number of warmup iterations.
    thin : int
        Thinning interval.
    proposal_scale_group_location : float
        Proposal scale for group-location coordinates.
    proposal_scale_group_log_scale : float
        Proposal scale for group-log-scale coordinates.
    proposal_scale_block_z : float
        Proposal scale for block-level unconstrained coordinates.
    random_seed : int | None
        Optional RNG seed.
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
    n_samples: int = 1000
    n_warmup: int = 500
    thin: int = 1
    proposal_scale_group_location: float = 0.08
    proposal_scale_group_log_scale: float = 0.05
    proposal_scale_block_z: float = 0.08
    random_seed: int | None = None


@dataclass(frozen=True, slots=True)
class HierarchicalStanEstimatorSpec:
    """Parsed estimator spec for within-subject hierarchical Stan NUTS.

    Parameters
    ----------
    parameter_names : tuple[str, ...]
        Names of pooled parameters across blocks.
    transform_kinds : dict[str, str] | None
        Optional per-parameter transform kinds in
        ``{"identity", "unit_interval_logit", "positive_log"}``.
    initial_group_location : dict[str, float] | None
        Optional initial constrained group-location values.
    initial_group_scale : dict[str, float] | None
        Optional initial positive group-scale values.
    initial_block_params : tuple[dict[str, float], ...] | None
        Optional per-block initial constrained parameter values.
    initial_block_params_by_subject : dict[str, tuple[dict[str, float], ...]] | None
        Optional subject-specific per-block initial parameter mappings.
    mu_prior_mean : float | dict[str, float]
        Group-location prior mean in latent space. A mapping applies
        parameter-specific values by name.
    mu_prior_std : float | dict[str, float]
        Group-location prior standard deviation in latent space. A mapping
        applies parameter-specific values by name.
    log_sigma_prior_mean : float | dict[str, float]
        Group log-scale prior mean. A mapping applies parameter-specific values
        by name.
    log_sigma_prior_std : float | dict[str, float]
        Group log-scale prior standard deviation. A mapping applies
        parameter-specific values by name.
    n_samples : int
        Number of post-warmup draws per chain.
    n_warmup : int
        Number of warmup iterations per chain.
    thin : int
        Thinning interval.
    n_chains : int
        Number of Stan chains.
    parallel_chains : int | None
        Optional parallel chain count.
    adapt_delta : float
        NUTS target acceptance statistic in ``(0, 1)``.
    max_treedepth : int
        NUTS maximum tree depth.
    step_size : float | None
        Optional initial step size.
    refresh : int
        CmdStan progress refresh interval.
    random_seed : int | None
        Optional RNG seed.
    """

    parameter_names: tuple[str, ...]
    transform_kinds: dict[str, str] | None = None
    initial_group_location: dict[str, float] | None = None
    initial_group_scale: dict[str, float] | None = None
    initial_block_params: tuple[dict[str, float], ...] | None = None
    initial_block_params_by_subject: dict[str, tuple[dict[str, float], ...]] | None = None
    mu_prior_mean: float | dict[str, float] = 0.0
    mu_prior_std: float | dict[str, float] = 2.0
    log_sigma_prior_mean: float | dict[str, float] = -1.0
    log_sigma_prior_std: float | dict[str, float] = 1.0
    n_samples: int = 1000
    n_warmup: int = 500
    thin: int = 1
    n_chains: int = 4
    parallel_chains: int | None = None
    adapt_delta: float = 0.9
    max_treedepth: int = 12
    step_size: float | None = None
    refresh: int = 0
    random_seed: int | None = None


def mcmc_estimator_spec_from_config(estimator_cfg: Mapping[str, Any]) -> MCMCEstimatorSpec:
    """Parse MCMC estimator config mapping into :class:`MCMCEstimatorSpec`.

    Parameters
    ----------
    estimator_cfg : Mapping[str, Any]
        Estimator config mapping.

    Returns
    -------
    MCMCEstimatorSpec
        Parsed MCMC estimator specification.

    Raises
    ------
    ValueError
        If required fields are missing or invalid.
    """

    estimator = _require_mapping(estimator_cfg, field_name="estimator")
    estimator_type = _coerce_non_empty_str(estimator.get("type"), field_name="estimator.type")
    if estimator_type != "random_walk_metropolis":
        raise ValueError("estimator.type must be 'random_walk_metropolis' for MCMC config")
    validate_allowed_keys(
        estimator,
        field_name="estimator",
        allowed_keys=(
            "type",
            "initial_params",
            "n_samples",
            "n_warmup",
            "thin",
            "proposal_scales",
            "bounds",
            "random_seed",
        ),
    )

    n_samples = int(estimator.get("n_samples", 0))
    if n_samples <= 0:
        raise ValueError("estimator.n_samples must be > 0")

    n_warmup = int(estimator.get("n_warmup", 500))
    if n_warmup < 0:
        raise ValueError("estimator.n_warmup must be >= 0")

    thin = int(estimator.get("thin", 1))
    if thin <= 0:
        raise ValueError("estimator.thin must be > 0")

    return MCMCEstimatorSpec(
        initial_params=_coerce_float_mapping(
            estimator.get("initial_params"),
            field_name="estimator.initial_params",
        ),
        n_samples=n_samples,
        n_warmup=n_warmup,
        thin=thin,
        proposal_scales=(
            _coerce_float_mapping(
                estimator.get("proposal_scales"),
                field_name="estimator.proposal_scales",
            )
            if estimator.get("proposal_scales") is not None
            else None
        ),
        bounds=(
            _coerce_bounds_mapping(estimator.get("bounds"), field_name="estimator.bounds")
            if estimator.get("bounds") is not None
            else None
        ),
        random_seed=(
            int(estimator["random_seed"])
            if estimator.get("random_seed") is not None
            else None
        ),
    )


def hierarchical_mcmc_estimator_spec_from_config(
    estimator_cfg: Mapping[str, Any],
) -> HierarchicalMCMCEstimatorSpec:
    """Parse hierarchical MCMC estimator config mapping.

    Parameters
    ----------
    estimator_cfg : Mapping[str, Any]
        Estimator config mapping.

    Returns
    -------
    HierarchicalMCMCEstimatorSpec
        Parsed hierarchical MCMC estimator specification.
    """

    estimator = _require_mapping(estimator_cfg, field_name="estimator")
    estimator_type = _coerce_non_empty_str(estimator.get("type"), field_name="estimator.type")
    if estimator_type != "within_subject_hierarchical_random_walk_metropolis":
        raise ValueError(
            "estimator.type must be "
            "'within_subject_hierarchical_random_walk_metropolis' "
            "for hierarchical MCMC config"
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
            "n_samples",
            "n_warmup",
            "thin",
            "proposal_scale_group_location",
            "proposal_scale_group_log_scale",
            "proposal_scale_block_z",
            "random_seed",
        ),
    )

    raw_names = _require_sequence(estimator.get("parameter_names"), field_name="estimator.parameter_names")
    parameter_names = tuple(
        _coerce_non_empty_str(name, field_name=f"estimator.parameter_names[{index}]")
        for index, name in enumerate(raw_names)
    )
    if len(set(parameter_names)) != len(parameter_names):
        raise ValueError("estimator.parameter_names must be unique")

    n_samples = int(estimator.get("n_samples", 0))
    if n_samples <= 0:
        raise ValueError("estimator.n_samples must be > 0")
    n_warmup = int(estimator.get("n_warmup", 500))
    if n_warmup < 0:
        raise ValueError("estimator.n_warmup must be >= 0")
    thin = int(estimator.get("thin", 1))
    if thin <= 0:
        raise ValueError("estimator.thin must be > 0")

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

    return HierarchicalMCMCEstimatorSpec(
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
        n_samples=n_samples,
        n_warmup=n_warmup,
        thin=thin,
        proposal_scale_group_location=float(estimator.get("proposal_scale_group_location", 0.08)),
        proposal_scale_group_log_scale=float(estimator.get("proposal_scale_group_log_scale", 0.05)),
        proposal_scale_block_z=float(estimator.get("proposal_scale_block_z", 0.08)),
        random_seed=(
            int(estimator["random_seed"])
            if estimator.get("random_seed") is not None
            else None
        ),
    )


def hierarchical_stan_estimator_spec_from_config(
    estimator_cfg: Mapping[str, Any],
) -> HierarchicalStanEstimatorSpec:
    """Parse hierarchical Stan NUTS estimator config mapping.

    Parameters
    ----------
    estimator_cfg : Mapping[str, Any]
        Estimator config mapping.

    Returns
    -------
    HierarchicalStanEstimatorSpec
        Parsed hierarchical Stan estimator specification.
    """

    estimator = _require_mapping(estimator_cfg, field_name="estimator")
    estimator_type = _coerce_non_empty_str(estimator.get("type"), field_name="estimator.type")
    if estimator_type != "within_subject_hierarchical_stan_nuts":
        raise ValueError(
            "estimator.type must be 'within_subject_hierarchical_stan_nuts' "
            "for hierarchical Stan config"
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
            "n_samples",
            "n_warmup",
            "thin",
            "n_chains",
            "parallel_chains",
            "adapt_delta",
            "max_treedepth",
            "step_size",
            "refresh",
            "random_seed",
        ),
    )

    raw_names = _require_sequence(estimator.get("parameter_names"), field_name="estimator.parameter_names")
    parameter_names = tuple(
        _coerce_non_empty_str(name, field_name=f"estimator.parameter_names[{index}]")
        for index, name in enumerate(raw_names)
    )
    if len(set(parameter_names)) != len(parameter_names):
        raise ValueError("estimator.parameter_names must be unique")

    n_samples = int(estimator.get("n_samples", 0))
    if n_samples <= 0:
        raise ValueError("estimator.n_samples must be > 0")
    n_warmup = int(estimator.get("n_warmup", 500))
    if n_warmup < 0:
        raise ValueError("estimator.n_warmup must be >= 0")
    thin = int(estimator.get("thin", 1))
    if thin <= 0:
        raise ValueError("estimator.thin must be > 0")
    n_chains = int(estimator.get("n_chains", 4))
    if n_chains <= 0:
        raise ValueError("estimator.n_chains must be > 0")

    parallel_chains = (
        int(estimator["parallel_chains"])
        if estimator.get("parallel_chains") is not None
        else None
    )
    if parallel_chains is not None and parallel_chains <= 0:
        raise ValueError("estimator.parallel_chains must be > 0")

    adapt_delta = float(estimator.get("adapt_delta", 0.9))
    if adapt_delta <= 0.0 or adapt_delta >= 1.0:
        raise ValueError("estimator.adapt_delta must be in (0, 1)")

    max_treedepth = int(estimator.get("max_treedepth", 12))
    if max_treedepth <= 0:
        raise ValueError("estimator.max_treedepth must be > 0")

    refresh = int(estimator.get("refresh", 0))
    if refresh < 0:
        raise ValueError("estimator.refresh must be >= 0")

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

    return HierarchicalStanEstimatorSpec(
        parameter_names=parameter_names,
        transform_kinds=(
            _parse_transform_kinds(estimator.get("transforms", {}), field_name="estimator.transforms")
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
        mu_prior_mean=_coerce_float_or_mapping(
            estimator.get("mu_prior_mean", 0.0),
            field_name="estimator.mu_prior_mean",
        ),
        mu_prior_std=_coerce_float_or_mapping(
            estimator.get("mu_prior_std", 2.0),
            field_name="estimator.mu_prior_std",
        ),
        log_sigma_prior_mean=_coerce_float_or_mapping(
            estimator.get("log_sigma_prior_mean", -1.0),
            field_name="estimator.log_sigma_prior_mean",
        ),
        log_sigma_prior_std=_coerce_float_or_mapping(
            estimator.get("log_sigma_prior_std", 1.0),
            field_name="estimator.log_sigma_prior_std",
        ),
        n_samples=n_samples,
        n_warmup=n_warmup,
        thin=thin,
        n_chains=n_chains,
        parallel_chains=parallel_chains,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
        step_size=(float(estimator["step_size"]) if estimator.get("step_size") is not None else None),
        refresh=refresh,
        random_seed=(
            int(estimator["random_seed"])
            if estimator.get("random_seed") is not None
            else None
        ),
    )


def sample_posterior_dataset_from_config(
    data: EpisodeTrace | BlockData | tuple[TrialDecision, ...] | list[TrialDecision],
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> MCMCPosteriorResult:
    """Sample posterior draws from config for one dataset.

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
    MCMCPosteriorResult
        Posterior sampling output.
    """

    cfg = _require_mapping(config, field_name="config")
    validate_allowed_keys(
        cfg,
        field_name="config",
        allowed_keys=("model", "prior", "estimator", "likelihood"),
    )
    model_spec = model_component_spec_from_config(
        _require_mapping(cfg.get("model"), field_name="config.model")
    )
    prior_program: PriorProgram = prior_program_from_config(
        _require_mapping(cfg.get("prior"), field_name="config.prior")
    )
    estimator_spec = mcmc_estimator_spec_from_config(
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
    return sample_posterior_model_from_registry(
        data,
        model_component_id=model_spec.component_id,
        prior_program=prior_program,
        initial_params=estimator_spec.initial_params,
        n_samples=estimator_spec.n_samples,
        n_warmup=estimator_spec.n_warmup,
        thin=estimator_spec.thin,
        proposal_scales=estimator_spec.proposal_scales,
        bounds=estimator_spec.bounds,
        model_kwargs=model_spec.kwargs,
        registry=reg,
        likelihood_program=resolved_likelihood,
        random_seed=estimator_spec.random_seed,
    )


def sample_posterior_block_from_config(
    block: BlockData,
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> MCMCBlockResult:
    """Sample posterior draws from config for one block.

    Parameters
    ----------
    block : BlockData
        Block dataset.
    config : Mapping[str, Any]
        Config with ``model``, ``prior``, ``estimator``, and optional
        ``likelihood`` sections.
    registry : PluginRegistry | None, optional
        Optional plugin registry.
    likelihood_program : LikelihoodProgram | None, optional
        Explicit likelihood evaluator override.
    """

    cfg = _require_mapping(config, field_name="config")
    validate_allowed_keys(
        cfg,
        field_name="config",
        allowed_keys=("model", "prior", "estimator", "likelihood"),
    )
    model_spec = model_component_spec_from_config(
        _require_mapping(cfg.get("model"), field_name="config.model")
    )
    prior_program: PriorProgram = prior_program_from_config(
        _require_mapping(cfg.get("prior"), field_name="config.prior")
    )
    estimator_spec = mcmc_estimator_spec_from_config(
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
    return sample_posterior_block_data(
        block,
        model_component_id=model_spec.component_id,
        prior_program=prior_program,
        initial_params=estimator_spec.initial_params,
        n_samples=estimator_spec.n_samples,
        n_warmup=estimator_spec.n_warmup,
        thin=estimator_spec.thin,
        proposal_scales=estimator_spec.proposal_scales,
        bounds=estimator_spec.bounds,
        model_kwargs=model_spec.kwargs,
        registry=reg,
        likelihood_program=resolved_likelihood,
        random_seed=estimator_spec.random_seed,
    )


def sample_posterior_subject_from_config(
    subject: SubjectData,
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> MCMCSubjectResult:
    """Sample posterior draws from config for one subject.

    Parameters
    ----------
    subject : SubjectData
        Subject dataset.
    config : Mapping[str, Any]
        Config with ``model``, ``prior``, ``estimator``, and optional
        ``likelihood`` sections.
    registry : PluginRegistry | None, optional
        Optional plugin registry.
    likelihood_program : LikelihoodProgram | None, optional
        Explicit likelihood evaluator override.
    """

    cfg = _require_mapping(config, field_name="config")
    validate_allowed_keys(
        cfg,
        field_name="config",
        allowed_keys=("model", "prior", "estimator", "likelihood"),
    )
    model_spec = model_component_spec_from_config(
        _require_mapping(cfg.get("model"), field_name="config.model")
    )
    prior_program: PriorProgram = prior_program_from_config(
        _require_mapping(cfg.get("prior"), field_name="config.prior")
    )
    estimator_spec = mcmc_estimator_spec_from_config(
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
    return sample_posterior_subject_data(
        subject,
        model_component_id=model_spec.component_id,
        prior_program=prior_program,
        initial_params=estimator_spec.initial_params,
        n_samples=estimator_spec.n_samples,
        n_warmup=estimator_spec.n_warmup,
        thin=estimator_spec.thin,
        proposal_scales=estimator_spec.proposal_scales,
        bounds=estimator_spec.bounds,
        model_kwargs=model_spec.kwargs,
        registry=reg,
        likelihood_program=resolved_likelihood,
        random_seed=estimator_spec.random_seed,
    )


def sample_posterior_study_from_config(
    study: StudyData,
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> MCMCStudyResult:
    """Sample posterior draws from config for one study.

    Parameters
    ----------
    study : StudyData
        Study dataset.
    config : Mapping[str, Any]
        Config with ``model``, ``prior``, ``estimator``, and optional
        ``likelihood`` sections.
    registry : PluginRegistry | None, optional
        Optional plugin registry.
    likelihood_program : LikelihoodProgram | None, optional
        Explicit likelihood evaluator override.
    """

    cfg = _require_mapping(config, field_name="config")
    validate_allowed_keys(
        cfg,
        field_name="config",
        allowed_keys=("model", "prior", "estimator", "likelihood"),
    )
    model_spec = model_component_spec_from_config(
        _require_mapping(cfg.get("model"), field_name="config.model")
    )
    prior_program: PriorProgram = prior_program_from_config(
        _require_mapping(cfg.get("prior"), field_name="config.prior")
    )
    estimator_spec = mcmc_estimator_spec_from_config(
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
    return sample_posterior_study_data(
        study,
        model_component_id=model_spec.component_id,
        prior_program=prior_program,
        initial_params=estimator_spec.initial_params,
        n_samples=estimator_spec.n_samples,
        n_warmup=estimator_spec.n_warmup,
        thin=estimator_spec.thin,
        proposal_scales=estimator_spec.proposal_scales,
        bounds=estimator_spec.bounds,
        model_kwargs=model_spec.kwargs,
        registry=reg,
        likelihood_program=resolved_likelihood,
        random_seed=estimator_spec.random_seed,
    )


def sample_subject_hierarchical_posterior_from_config(
    subject: SubjectData,
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> HierarchicalSubjectPosteriorResult:
    """Sample hierarchical posterior draws for one subject from config.

    Parameters
    ----------
    subject : SubjectData
        Subject dataset.
    config : Mapping[str, Any]
        Config with ``model``, ``estimator``, and optional ``likelihood``.
    registry : PluginRegistry | None, optional
        Optional plugin registry.
    likelihood_program : LikelihoodProgram | None, optional
        Explicit likelihood evaluator override.

    Returns
    -------
    HierarchicalSubjectPosteriorResult
        Subject-level hierarchical posterior output.
    """

    cfg = _require_mapping(config, field_name="config")
    validate_allowed_keys(
        cfg,
        field_name="config",
        allowed_keys=("model", "estimator", "likelihood"),
    )
    model_spec = model_component_spec_from_config(
        _require_mapping(cfg.get("model"), field_name="config.model")
    )
    estimator_cfg = _require_mapping(cfg.get("estimator"), field_name="config.estimator")
    estimator_type = _coerce_non_empty_str(estimator_cfg.get("type"), field_name="config.estimator.type")
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

    if estimator_type == "within_subject_hierarchical_random_walk_metropolis":
        mcmc_spec = hierarchical_mcmc_estimator_spec_from_config(estimator_cfg)
        fixed_kwargs = dict(model_spec.kwargs)
        model_factory = lambda params: reg.create_model(
            model_spec.component_id,
            **_merge_kwargs(fixed_kwargs, params),
        )
        return sample_subject_hierarchical_posterior(
            subject,
            model_factory=model_factory,
            parameter_names=mcmc_spec.parameter_names,
            transforms=mcmc_spec.transforms,
            likelihood_program=resolved_likelihood,
            requirements=manifest.requirements,
            initial_group_location=mcmc_spec.initial_group_location,
            initial_group_scale=mcmc_spec.initial_group_scale,
            initial_block_params=mcmc_spec.initial_block_params,
            mu_prior_mean=mcmc_spec.mu_prior_mean,
            mu_prior_std=mcmc_spec.mu_prior_std,
            log_sigma_prior_mean=mcmc_spec.log_sigma_prior_mean,
            log_sigma_prior_std=mcmc_spec.log_sigma_prior_std,
            n_samples=mcmc_spec.n_samples,
            n_warmup=mcmc_spec.n_warmup,
            thin=mcmc_spec.thin,
            proposal_scale_group_location=mcmc_spec.proposal_scale_group_location,
            proposal_scale_group_log_scale=mcmc_spec.proposal_scale_group_log_scale,
            proposal_scale_block_z=mcmc_spec.proposal_scale_block_z,
            random_seed=mcmc_spec.random_seed,
        )

    if estimator_type == "within_subject_hierarchical_stan_nuts":
        stan_spec = hierarchical_stan_estimator_spec_from_config(estimator_cfg)
        return sample_subject_hierarchical_posterior_stan(
            subject,
            model_component_id=model_spec.component_id,
            model_kwargs=model_spec.kwargs,
            parameter_names=stan_spec.parameter_names,
            transform_kinds=stan_spec.transform_kinds,
            requirements=manifest.requirements,
            initial_group_location=stan_spec.initial_group_location,
            initial_group_scale=stan_spec.initial_group_scale,
            initial_block_params=stan_spec.initial_block_params,
            mu_prior_mean=stan_spec.mu_prior_mean,
            mu_prior_std=stan_spec.mu_prior_std,
            log_sigma_prior_mean=stan_spec.log_sigma_prior_mean,
            log_sigma_prior_std=stan_spec.log_sigma_prior_std,
            n_samples=stan_spec.n_samples,
            n_warmup=stan_spec.n_warmup,
            thin=stan_spec.thin,
            n_chains=stan_spec.n_chains,
            parallel_chains=stan_spec.parallel_chains,
            adapt_delta=stan_spec.adapt_delta,
            max_treedepth=stan_spec.max_treedepth,
            step_size=stan_spec.step_size,
            random_seed=stan_spec.random_seed,
            refresh=stan_spec.refresh,
        )

    raise ValueError(
        f"unsupported hierarchical estimator.type {estimator_type!r}; "
        "expected one of ['within_subject_hierarchical_random_walk_metropolis', "
        "'within_subject_hierarchical_stan_nuts']"
    )


def sample_study_hierarchical_posterior_from_config(
    study: StudyData,
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> HierarchicalStudyPosteriorResult:
    """Sample hierarchical posterior draws for all study subjects from config.

    Parameters
    ----------
    study : StudyData
        Study dataset.
    config : Mapping[str, Any]
        Config with ``model``, ``estimator``, and optional ``likelihood``.
    registry : PluginRegistry | None, optional
        Optional plugin registry.
    likelihood_program : LikelihoodProgram | None, optional
        Explicit likelihood evaluator override.

    Returns
    -------
    HierarchicalStudyPosteriorResult
        Study-level hierarchical posterior output.
    """

    cfg = _require_mapping(config, field_name="config")
    validate_allowed_keys(
        cfg,
        field_name="config",
        allowed_keys=("model", "estimator", "likelihood"),
    )
    model_spec = model_component_spec_from_config(
        _require_mapping(cfg.get("model"), field_name="config.model")
    )
    estimator_cfg = _require_mapping(cfg.get("estimator"), field_name="config.estimator")
    estimator_type = _coerce_non_empty_str(estimator_cfg.get("type"), field_name="config.estimator.type")
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

    if estimator_type == "within_subject_hierarchical_random_walk_metropolis":
        mcmc_spec = hierarchical_mcmc_estimator_spec_from_config(estimator_cfg)
        fixed_kwargs = dict(model_spec.kwargs)
        model_factory = lambda params: reg.create_model(
            model_spec.component_id,
            **_merge_kwargs(fixed_kwargs, params),
        )
        return sample_study_hierarchical_posterior(
            study,
            model_factory=model_factory,
            parameter_names=mcmc_spec.parameter_names,
            transforms=mcmc_spec.transforms,
            likelihood_program=resolved_likelihood,
            requirements=manifest.requirements,
            initial_group_location=mcmc_spec.initial_group_location,
            initial_group_scale=mcmc_spec.initial_group_scale,
            initial_block_params_by_subject=mcmc_spec.initial_block_params_by_subject,
            mu_prior_mean=mcmc_spec.mu_prior_mean,
            mu_prior_std=mcmc_spec.mu_prior_std,
            log_sigma_prior_mean=mcmc_spec.log_sigma_prior_mean,
            log_sigma_prior_std=mcmc_spec.log_sigma_prior_std,
            n_samples=mcmc_spec.n_samples,
            n_warmup=mcmc_spec.n_warmup,
            thin=mcmc_spec.thin,
            proposal_scale_group_location=mcmc_spec.proposal_scale_group_location,
            proposal_scale_group_log_scale=mcmc_spec.proposal_scale_group_log_scale,
            proposal_scale_block_z=mcmc_spec.proposal_scale_block_z,
            random_seed=mcmc_spec.random_seed,
        )

    if estimator_type == "within_subject_hierarchical_stan_nuts":
        stan_spec = hierarchical_stan_estimator_spec_from_config(estimator_cfg)
        return sample_study_hierarchical_posterior_stan(
            study,
            model_component_id=model_spec.component_id,
            model_kwargs=model_spec.kwargs,
            parameter_names=stan_spec.parameter_names,
            transform_kinds=stan_spec.transform_kinds,
            requirements=manifest.requirements,
            initial_group_location=stan_spec.initial_group_location,
            initial_group_scale=stan_spec.initial_group_scale,
            initial_block_params_by_subject=stan_spec.initial_block_params_by_subject,
            mu_prior_mean=stan_spec.mu_prior_mean,
            mu_prior_std=stan_spec.mu_prior_std,
            log_sigma_prior_mean=stan_spec.log_sigma_prior_mean,
            log_sigma_prior_std=stan_spec.log_sigma_prior_std,
            n_samples=stan_spec.n_samples,
            n_warmup=stan_spec.n_warmup,
            thin=stan_spec.thin,
            n_chains=stan_spec.n_chains,
            parallel_chains=stan_spec.parallel_chains,
            adapt_delta=stan_spec.adapt_delta,
            max_treedepth=stan_spec.max_treedepth,
            step_size=stan_spec.step_size,
            random_seed=stan_spec.random_seed,
            refresh=stan_spec.refresh,
        )

    raise ValueError(
        f"unsupported hierarchical estimator.type {estimator_type!r}; "
        "expected one of ['within_subject_hierarchical_random_walk_metropolis', "
        "'within_subject_hierarchical_stan_nuts']"
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


def _parse_transform_kinds(raw: Any, *, field_name: str) -> dict[str, str]:
    """Parse configured transform-kind mapping for Stan estimators."""

    mapping = _require_mapping(raw, field_name=field_name)
    out: dict[str, str] = {}
    for param_name, spec in mapping.items():
        if isinstance(spec, str):
            kind = str(spec).strip()
        else:
            spec_mapping = _require_mapping(spec, field_name=f"{field_name}.{param_name}")
            validate_allowed_keys(
                spec_mapping,
                field_name=f"{field_name}.{param_name}",
                allowed_keys=("kind",),
            )
            kind = _coerce_non_empty_str(
                spec_mapping.get("kind"),
                field_name=f"{field_name}.{param_name}.kind",
            )

        if kind not in {"identity", "unit_interval_logit", "positive_log"}:
            raise ValueError(
                f"unsupported transform {kind!r}; expected one of "
                "{'identity', 'unit_interval_logit', 'positive_log'}"
            )
        out[str(param_name)] = kind
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


def _merge_kwargs(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge fixed keyword arguments with free-parameter overrides."""

    merged = dict(base)
    merged.update(dict(override))
    return merged


def _coerce_bounds_mapping(
    raw: Any,
    *,
    field_name: str,
) -> dict[str, tuple[float | None, float | None]]:
    """Parse bounds mapping from config."""

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


def _coerce_float_or_mapping(raw: Any, *, field_name: str) -> float | dict[str, float]:
    """Coerce one numeric prior spec as float or mapping of floats."""

    if isinstance(raw, Mapping):
        return _coerce_float_mapping(raw, field_name=field_name)
    return float(raw)


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
    "HierarchicalMCMCEstimatorSpec",
    "HierarchicalStanEstimatorSpec",
    "MCMCEstimatorSpec",
    "hierarchical_mcmc_estimator_spec_from_config",
    "hierarchical_stan_estimator_spec_from_config",
    "mcmc_estimator_spec_from_config",
    "sample_posterior_block_from_config",
    "sample_posterior_dataset_from_config",
    "sample_posterior_study_from_config",
    "sample_posterior_subject_from_config",
    "sample_study_hierarchical_posterior_from_config",
    "sample_subject_hierarchical_posterior_from_config",
]
