"""Config-driven Stan hierarchical posterior sampling helpers.

Pure-Python MCMC samplers have been removed. Bayesian posterior sampling is
Stan-backed via ``within_subject_hierarchical_stan_nuts``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from comp_model.core.config_validation import validate_allowed_keys
from comp_model.core.data import StudyData, SubjectData
from comp_model.plugins import PluginRegistry, build_default_registry

from .config import model_component_spec_from_config
from .hierarchical_mcmc import (
    HierarchicalStudyPosteriorResult,
    HierarchicalSubjectPosteriorResult,
)
from .hierarchical_stan import (
    sample_study_hierarchical_posterior_stan,
    sample_subject_hierarchical_posterior_stan,
)


@dataclass(frozen=True, slots=True)
class HierarchicalStanEstimatorSpec:
    """Parsed estimator spec for within-subject hierarchical Stan NUTS."""

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


def hierarchical_stan_estimator_spec_from_config(
    estimator_cfg: Mapping[str, Any],
) -> HierarchicalStanEstimatorSpec:
    """Parse hierarchical Stan NUTS estimator config mapping."""

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


def sample_subject_hierarchical_posterior_from_config(
    subject: SubjectData,
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
) -> HierarchicalSubjectPosteriorResult:
    """Sample hierarchical posterior draws for one subject from config."""

    cfg = _require_mapping(config, field_name="config")
    validate_allowed_keys(
        cfg,
        field_name="config",
        allowed_keys=("model", "estimator"),
    )
    model_spec = model_component_spec_from_config(
        _require_mapping(cfg.get("model"), field_name="config.model")
    )
    estimator_cfg = _require_mapping(cfg.get("estimator"), field_name="config.estimator")
    stan_spec = hierarchical_stan_estimator_spec_from_config(estimator_cfg)

    reg = registry if registry is not None else build_default_registry()
    manifest = reg.get("model", model_spec.component_id)
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


def sample_study_hierarchical_posterior_from_config(
    study: StudyData,
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
) -> HierarchicalStudyPosteriorResult:
    """Sample hierarchical posterior draws for all study subjects from config."""

    cfg = _require_mapping(config, field_name="config")
    validate_allowed_keys(
        cfg,
        field_name="config",
        allowed_keys=("model", "estimator"),
    )
    model_spec = model_component_spec_from_config(
        _require_mapping(cfg.get("model"), field_name="config.model")
    )
    estimator_cfg = _require_mapping(cfg.get("estimator"), field_name="config.estimator")
    stan_spec = hierarchical_stan_estimator_spec_from_config(estimator_cfg)

    reg = registry if registry is not None else build_default_registry()
    manifest = reg.get("model", model_spec.component_id)
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
    "HierarchicalStanEstimatorSpec",
    "hierarchical_stan_estimator_spec_from_config",
    "sample_study_hierarchical_posterior_from_config",
    "sample_subject_hierarchical_posterior_from_config",
]
