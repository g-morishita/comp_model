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
from comp_model.plugins import PluginRegistry

from .bayes import (
    BayesFitResult,
    IndependentPriorProgram,
    MapFitSpec,
    PriorProgram,
    beta_log_prior,
    log_normal_log_prior,
    normal_log_prior,
    uniform_log_prior,
)
from .hierarchical import (
    HierarchicalStudyMapResult,
    HierarchicalSubjectMapResult,
)
from .likelihood import LikelihoodProgram
from .map_study_fitting import (
    MapBlockFitResult,
    MapStudyFitResult,
    MapSubjectFitResult,
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

    raise ValueError(
        "SciPy MAP estimators are no longer supported. "
        "Use estimator.type='within_subject_hierarchical_stan_map'."
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

    raise ValueError(
        "within_subject_hierarchical_map has been removed. "
        "Use estimator.type='within_subject_hierarchical_stan_map'."
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

    raise RuntimeError(
        "fit_map_dataset_from_config has been removed. "
        "Use fit_dataset_auto_from_config with estimator.type='within_subject_hierarchical_stan_map'."
    )


def fit_map_block_from_config(
    block: BlockData,
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> MapBlockFitResult:
    """Fit one block with MAP using declarative config."""

    raise RuntimeError(
        "fit_map_block_from_config has been removed. "
        "Use fit_block_auto_from_config with estimator.type='within_subject_hierarchical_stan_map'."
    )


def fit_map_subject_from_config(
    subject: SubjectData,
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> MapSubjectFitResult:
    """Fit one subject with MAP using declarative config."""

    raise RuntimeError(
        "fit_map_subject_from_config has been removed. "
        "Use fit_subject_auto_from_config with estimator.type='within_subject_hierarchical_stan_map'."
    )


def fit_map_study_from_config(
    study: StudyData,
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> MapStudyFitResult:
    """Fit one study with MAP using declarative config."""

    raise RuntimeError(
        "fit_map_study_from_config has been removed. "
        "Use fit_study_auto_from_config with estimator.type='within_subject_hierarchical_stan_map'."
    )


def fit_subject_hierarchical_map_from_config(
    subject: SubjectData,
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> HierarchicalSubjectMapResult:
    """Removed legacy hierarchical MAP config entry point.

    Parameters
    ----------
    subject : SubjectData
        Subject dataset.
    config : Mapping[str, Any]
        Legacy config mapping.
    registry : PluginRegistry | None, optional
        Optional plugin registry.
    likelihood_program : LikelihoodProgram | None, optional
        Optional likelihood evaluator.

    Returns
    -------
    HierarchicalSubjectMapResult
        Unused legacy return type.
    """

    raise RuntimeError(
        "fit_subject_hierarchical_map_from_config has been removed. "
        "Use sample_subject_hierarchical_posterior_from_config with "
        "estimator.type='within_subject_hierarchical_stan_map'."
    )


def fit_study_hierarchical_map_from_config(
    study: StudyData,
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> HierarchicalStudyMapResult:
    """Removed legacy hierarchical MAP config entry point.

    Parameters
    ----------
    study : StudyData
        Study dataset.
    config : Mapping[str, Any]
        Legacy config mapping.
    registry : PluginRegistry | None, optional
        Optional plugin registry.
    likelihood_program : LikelihoodProgram | None, optional
        Optional likelihood evaluator.

    Returns
    -------
    HierarchicalStudyMapResult
        Unused legacy return type.
    """

    raise RuntimeError(
        "fit_study_hierarchical_map_from_config has been removed. "
        "Use sample_study_hierarchical_posterior_from_config with "
        "estimator.type='within_subject_hierarchical_stan_map'."
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
