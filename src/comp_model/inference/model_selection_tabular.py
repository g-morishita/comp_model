"""Config-driven model-comparison helpers for tabular CSV datasets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from comp_model.core.data import SubjectData
from comp_model.io import read_study_decisions_csv, read_trial_decisions_csv
from comp_model.plugins import PluginRegistry

from .likelihood import LikelihoodProgram
from .model_selection import ModelComparisonResult
from .model_selection_config import (
    compare_dataset_candidates_from_config,
    compare_study_candidates_from_config,
    compare_subject_candidates_from_config,
)
from .study_model_selection import StudyModelComparisonResult, SubjectModelComparisonResult


def compare_trial_csv_candidates_from_config(
    path: str,
    *,
    config: Mapping[str, Any],
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> ModelComparisonResult:
    """Compare configured model candidates on one trial-level CSV dataset.

    Parameters
    ----------
    path : str
        Path to CSV written with ``write_trial_decisions_csv``.
    config : Mapping[str, Any]
        Declarative model-comparison config containing ``candidates`` and
        optional criterion/likelihood sections.
    registry : PluginRegistry | None, optional
        Optional plugin registry.
    likelihood_program : LikelihoodProgram | None, optional
        Optional replay-likelihood implementation.

    Returns
    -------
    ModelComparisonResult
        Candidate comparison output for one dataset.
    """

    decisions = read_trial_decisions_csv(path)
    return compare_dataset_candidates_from_config(
        decisions,
        config=config,
        registry=registry,
        likelihood_program=likelihood_program,
    )


def compare_study_csv_candidates_from_config(
    path: str,
    *,
    config: Mapping[str, Any],
    level: Literal["study", "subject"] = "study",
    subject_id: str | None = None,
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> StudyModelComparisonResult | SubjectModelComparisonResult:
    """Compare configured candidates on study CSV data.

    Parameters
    ----------
    path : str
        Path to CSV written with ``write_study_decisions_csv``.
    config : Mapping[str, Any]
        Declarative model-comparison config containing ``candidates`` and
        optional criterion/likelihood sections.
    level : {"study", "subject"}, optional
        Aggregation level:
        ``"study"`` compares candidates after aggregating across subjects;
        ``"subject"`` compares candidates for exactly one resolved subject.
    subject_id : str | None, optional
        Subject selection key used when ``level="subject"``. If omitted,
        single-subject studies are selected implicitly.
    registry : PluginRegistry | None, optional
        Optional plugin registry.
    likelihood_program : LikelihoodProgram | None, optional
        Optional replay-likelihood implementation.

    Returns
    -------
    StudyModelComparisonResult | SubjectModelComparisonResult
        Candidate comparison output at the requested level.

    Raises
    ------
    ValueError
        If ``level`` is unsupported or subject selection fails.
    """

    study = read_study_decisions_csv(path)
    if level == "study":
        return compare_study_candidates_from_config(
            study,
            config=config,
            registry=registry,
            likelihood_program=likelihood_program,
        )
    if level != "subject":
        raise ValueError("level must be one of {'study', 'subject'}")

    subject = _resolve_subject(study.subjects, subject_id=subject_id)
    return compare_subject_candidates_from_config(
        subject,
        config=config,
        registry=registry,
        likelihood_program=likelihood_program,
    )


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


__all__ = [
    "compare_study_csv_candidates_from_config",
    "compare_trial_csv_candidates_from_config",
]
