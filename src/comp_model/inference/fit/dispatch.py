"""Auto-dispatch config fitting helpers.

These helpers route declarative fitting configs to MLE or Stan-backed Bayesian
estimators based on ``config.estimator.type``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from comp_model.core.data import BlockData, StudyData, SubjectData, TrialDecision
from comp_model.core.events import EpisodeTrace
from comp_model.plugins import PluginRegistry

from .config import (
    fit_block_from_config,
    fit_trace_from_config,
    fit_study_from_config,
    fit_subject_from_config,
)
from .core import coerce_episode_trace
from ..stan_config import STUDY_STAN_ESTIMATORS, SUBJECT_STAN_ESTIMATORS, infer_study_stan_from_config, infer_subject_stan_from_config

MLE_ESTIMATORS = {"mle"}
MAP_ESTIMATORS: set[str] = set()
MCMC_ESTIMATORS: set[str] = set()
SUBJECT_BAYES_ESTIMATORS = set(SUBJECT_STAN_ESTIMATORS)
STUDY_BAYES_ESTIMATORS = set(STUDY_STAN_ESTIMATORS)
BAYES_ESTIMATORS = SUBJECT_BAYES_ESTIMATORS | STUDY_BAYES_ESTIMATORS


def fit_trace_auto_from_config(
    data: EpisodeTrace | BlockData | tuple[TrialDecision, ...] | list[TrialDecision],
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
):
    """Fit one trace-like input by auto-dispatching on estimator type."""

    estimator_type = _estimator_type(config)
    if estimator_type in MLE_ESTIMATORS:
        return fit_trace_from_config(data, config=config, registry=registry)
    if estimator_type in SUBJECT_BAYES_ESTIMATORS:
        if isinstance(data, BlockData):
            block = data
        else:
            trace = coerce_episode_trace(data)
            block = BlockData(block_id="__trace__", event_trace=trace)
        subject = SubjectData(subject_id="__trace__", blocks=(block,))
        return infer_subject_stan_from_config(subject, config=config, registry=registry)
    if estimator_type in STUDY_BAYES_ESTIMATORS:
        raise ValueError(
            f"unsupported estimator.type {estimator_type!r} for trace fitting; "
            f"study-level Stan estimators require StudyData"
        )
    raise ValueError(
        f"unsupported estimator.type {estimator_type!r} for trace fitting; "
        f"expected one of {sorted(MLE_ESTIMATORS | BAYES_ESTIMATORS)}"
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
    if estimator_type in SUBJECT_BAYES_ESTIMATORS:
        wrapped_subject = SubjectData(subject_id="__block__", blocks=(block,))
        return infer_subject_stan_from_config(wrapped_subject, config=config, registry=registry)
    if estimator_type in STUDY_BAYES_ESTIMATORS:
        raise ValueError(
            f"unsupported estimator.type {estimator_type!r} for block fitting; "
            f"study-level Stan estimators require StudyData"
        )
    raise ValueError(
        f"unsupported estimator.type {estimator_type!r} for block fitting; "
        f"expected one of {sorted(MLE_ESTIMATORS | BAYES_ESTIMATORS)}"
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
    if estimator_type in SUBJECT_BAYES_ESTIMATORS:
        return infer_subject_stan_from_config(subject, config=config, registry=registry)
    if estimator_type in STUDY_BAYES_ESTIMATORS:
        raise ValueError(
            f"unsupported estimator.type {estimator_type!r} for subject fitting; "
            f"study-level Stan estimators require StudyData"
        )
    raise ValueError(
        f"unsupported estimator.type {estimator_type!r} for subject fitting; "
        f"expected one of {sorted(MLE_ESTIMATORS | BAYES_ESTIMATORS)}"
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
    if estimator_type in STUDY_BAYES_ESTIMATORS:
        return infer_study_stan_from_config(study, config=config, registry=registry)
    if estimator_type in SUBJECT_BAYES_ESTIMATORS:
        raise ValueError(
            f"unsupported estimator.type {estimator_type!r} for study fitting; "
            f"subject-level Stan estimators require SubjectData"
        )
    raise ValueError(
        f"unsupported estimator.type {estimator_type!r} for study fitting; "
        f"expected one of {sorted(MLE_ESTIMATORS | BAYES_ESTIMATORS)}"
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
    "BAYES_ESTIMATORS",
    "MAP_ESTIMATORS",
    "MCMC_ESTIMATORS",
    "MLE_ESTIMATORS",
    "STUDY_BAYES_ESTIMATORS",
    "SUBJECT_BAYES_ESTIMATORS",
    "fit_block_auto_from_config",
    "fit_trace_auto_from_config",
    "fit_study_auto_from_config",
    "fit_subject_auto_from_config",
]
