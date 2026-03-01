"""Config-driven fitting helpers for tabular CSV datasets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from comp_model.core.data import SubjectData
from comp_model.io import read_study_decisions_csv, read_trial_decisions_csv
from comp_model.plugins import PluginRegistry

from .config_dispatch import (
    fit_dataset_auto_from_config,
    fit_study_auto_from_config,
    fit_subject_auto_from_config,
)


def fit_trial_csv_from_config(
    path: str,
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
):
    """Fit one trial-level CSV dataset from declarative config.

    Parameters
    ----------
    path : str
        Path to CSV written with ``write_trial_decisions_csv``.
    config : Mapping[str, Any]
        Declarative fitting config containing ``estimator.type``.
    registry : PluginRegistry | None, optional
        Optional plugin registry.

    Returns
    -------
    Any
        Auto-dispatch fit result for one dataset.
    """

    decisions = read_trial_decisions_csv(path)
    return fit_dataset_auto_from_config(decisions, config=config, registry=registry)


def fit_study_csv_from_config(
    path: str,
    *,
    config: Mapping[str, Any],
    level: Literal["study", "subject"] = "study",
    subject_id: str | None = None,
    registry: PluginRegistry | None = None,
):
    """Fit a study/subject from flattened study CSV and config.

    Parameters
    ----------
    path : str
        Path to CSV written with ``write_study_decisions_csv``.
    config : Mapping[str, Any]
        Declarative fitting config containing ``estimator.type``.
    level : {"study", "subject"}, optional
        Fit aggregation level.
    subject_id : str | None, optional
        Subject ID used when ``level="subject"``. If omitted, single-subject
        studies are selected automatically.
    registry : PluginRegistry | None, optional
        Optional plugin registry.

    Returns
    -------
    Any
        Auto-dispatch fit result for the selected level.

    Raises
    ------
    ValueError
        If ``level`` is invalid or ``subject_id`` cannot be resolved.
    """

    study = read_study_decisions_csv(path)
    if level == "study":
        return fit_study_auto_from_config(study, config=config, registry=registry)
    if level != "subject":
        raise ValueError("level must be one of {'study', 'subject'}")

    subject = _resolve_subject(study.subjects, subject_id=subject_id)
    return fit_subject_auto_from_config(subject, config=config, registry=registry)


def _resolve_subject(subjects: tuple[SubjectData, ...], *, subject_id: str | None) -> SubjectData:
    """Resolve one subject from a study by explicit or implicit selection."""

    if subject_id is None:
        if len(subjects) != 1:
            raise ValueError("subject_id is required when study CSV includes multiple subjects")
        return subjects[0]

    for subject in subjects:
        if subject.subject_id == subject_id:
            return subject
    available = ", ".join(sorted(item.subject_id for item in subjects))
    raise ValueError(f"subject_id {subject_id!r} not found; available={available}")


__all__ = ["fit_study_csv_from_config", "fit_trial_csv_from_config"]
