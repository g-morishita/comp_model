"""Auto-dispatch config fitting helpers.

These helpers route declarative fitting configs to MLE, MAP, hierarchical MAP,
or hierarchical Stan posterior sampling based on ``config.estimator.type``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from comp_model.core.data import BlockData, StudyData, SubjectData, TrialDecision
from comp_model.core.events import EpisodeTrace
from comp_model.plugins import PluginRegistry

from .bayes_config import (
    fit_map_block_from_config,
    fit_map_dataset_from_config,
    fit_map_study_from_config,
    fit_map_subject_from_config,
    fit_study_hierarchical_map_from_config,
    fit_subject_hierarchical_map_from_config,
)
from .config import (
    fit_block_from_config,
    fit_dataset_from_config,
    fit_study_from_config,
    fit_subject_from_config,
)
from .mcmc_config import (
    sample_study_hierarchical_posterior_from_config,
    sample_subject_hierarchical_posterior_from_config,
)

MLE_ESTIMATORS = {"grid_search", "scipy_minimize", "transformed_scipy_minimize"}
MAP_ESTIMATORS = {"scipy_map", "transformed_scipy_map"}
MCMC_ESTIMATORS: set[str] = set()
HIERARCHICAL_MCMC_ESTIMATORS = {"within_subject_hierarchical_stan_nuts"}
HIERARCHICAL_ESTIMATORS = {"within_subject_hierarchical_map"}


def fit_dataset_auto_from_config(
    data: EpisodeTrace | BlockData | tuple[TrialDecision, ...] | list[TrialDecision],
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
):
    """Fit one dataset by auto-dispatching on estimator type.

    Parameters
    ----------
    data : EpisodeTrace | BlockData | tuple[TrialDecision, ...] | list[TrialDecision]
        Dataset container.
    config : Mapping[str, Any]
        Declarative config containing ``estimator.type``.
    registry : PluginRegistry | None, optional
        Optional plugin registry.

    Returns
    -------
    Any
        Fit result object (MLE, MAP, or MCMC), depending on estimator type.
    """

    estimator_type = _estimator_type(config)
    if estimator_type in MLE_ESTIMATORS:
        return fit_dataset_from_config(data, config=config, registry=registry)
    if estimator_type in MAP_ESTIMATORS:
        return fit_map_dataset_from_config(data, config=config, registry=registry)
    raise ValueError(
        f"unsupported estimator.type {estimator_type!r} for dataset fitting; "
        f"expected one of {sorted(MLE_ESTIMATORS | MAP_ESTIMATORS)}"
    )


def fit_block_auto_from_config(
    block: BlockData,
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
):
    """Fit one block by auto-dispatching on estimator type."""

    estimator_type = _estimator_type(config)
    if estimator_type in MLE_ESTIMATORS:
        return fit_block_from_config(block, config=config, registry=registry)
    if estimator_type in MAP_ESTIMATORS:
        return fit_map_block_from_config(block, config=config, registry=registry)
    raise ValueError(
        f"unsupported estimator.type {estimator_type!r} for block fitting; "
        f"expected one of {sorted(MLE_ESTIMATORS | MAP_ESTIMATORS)}"
    )


def fit_subject_auto_from_config(
    subject: SubjectData,
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
):
    """Fit one subject by auto-dispatching on estimator type."""

    estimator_type = _estimator_type(config)
    if estimator_type in MLE_ESTIMATORS:
        return fit_subject_from_config(subject, config=config, registry=registry)
    if estimator_type in MAP_ESTIMATORS:
        return fit_map_subject_from_config(subject, config=config, registry=registry)
    if estimator_type in HIERARCHICAL_ESTIMATORS:
        return fit_subject_hierarchical_map_from_config(
            subject,
            config=config,
            registry=registry,
        )
    if estimator_type in HIERARCHICAL_MCMC_ESTIMATORS:
        return sample_subject_hierarchical_posterior_from_config(
            subject,
            config=config,
            registry=registry,
        )
    raise ValueError(
        f"unsupported estimator.type {estimator_type!r} for subject fitting; "
        f"expected one of {sorted(MLE_ESTIMATORS | MAP_ESTIMATORS | HIERARCHICAL_ESTIMATORS | HIERARCHICAL_MCMC_ESTIMATORS)}"
    )


def fit_study_auto_from_config(
    study: StudyData,
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
):
    """Fit one study by auto-dispatching on estimator type."""

    estimator_type = _estimator_type(config)
    if estimator_type in MLE_ESTIMATORS:
        return fit_study_from_config(study, config=config, registry=registry)
    if estimator_type in MAP_ESTIMATORS:
        return fit_map_study_from_config(study, config=config, registry=registry)
    if estimator_type in HIERARCHICAL_ESTIMATORS:
        return fit_study_hierarchical_map_from_config(study, config=config, registry=registry)
    if estimator_type in HIERARCHICAL_MCMC_ESTIMATORS:
        return sample_study_hierarchical_posterior_from_config(
            study,
            config=config,
            registry=registry,
        )
    raise ValueError(
        f"unsupported estimator.type {estimator_type!r} for study fitting; "
        f"expected one of {sorted(MLE_ESTIMATORS | MAP_ESTIMATORS | HIERARCHICAL_ESTIMATORS | HIERARCHICAL_MCMC_ESTIMATORS)}"
    )


def _estimator_type(config: Mapping[str, Any]) -> str:
    """Read and validate ``config.estimator.type``."""

    if not isinstance(config, Mapping):
        raise ValueError("config must be an object")

    estimator = config.get("estimator")
    if not isinstance(estimator, Mapping):
        raise ValueError("config.estimator must be an object")

    raw_type = estimator.get("type")
    if raw_type is None:
        raise ValueError("config.estimator.type must be a non-empty string")
    value = str(raw_type).strip()
    if not value:
        raise ValueError("config.estimator.type must be a non-empty string")
    return value


__all__ = [
    "HIERARCHICAL_MCMC_ESTIMATORS",
    "HIERARCHICAL_ESTIMATORS",
    "MCMC_ESTIMATORS",
    "MAP_ESTIMATORS",
    "MLE_ESTIMATORS",
    "fit_block_auto_from_config",
    "fit_dataset_auto_from_config",
    "fit_study_auto_from_config",
    "fit_subject_auto_from_config",
]
