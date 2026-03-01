"""Tests for subject/study model-comparison serialization helpers."""

from __future__ import annotations

import csv
from pathlib import Path

from comp_model.core.data import BlockData, StudyData, SubjectData, TrialDecision
from comp_model.inference import (
    CandidateFitSpec,
    MLECandidate,
    MLEFitResult,
    compare_study_candidate_models,
)
from comp_model.inference.serialization import (
    study_model_comparison_records,
    study_model_comparison_subject_records,
    subject_model_comparison_records,
    write_study_model_comparison_csv,
    write_study_model_comparison_subject_csv,
    write_subject_model_comparison_csv,
)


def _trial(trial_index: int, action: int, reward: float) -> TrialDecision:
    """Build one trial row for study-model-comparison serialization tests."""

    return TrialDecision(
        trial_index=trial_index,
        decision_index=0,
        actor_id="subject",
        available_actions=(0, 1),
        action=action,
        observation={"state": 0},
        outcome={"reward": reward},
    )


def _constant_mle_fit(*, log_likelihood: float, alpha: float):
    """Build deterministic MLE fit function for testing."""

    def _fit(trace):
        candidate = MLECandidate(params={"alpha": alpha}, log_likelihood=log_likelihood)
        return MLEFitResult(best=candidate, candidates=(candidate,))

    return _fit


def _study_result():
    """Build one deterministic study model-comparison result."""

    study = StudyData(
        subjects=(
            SubjectData(
                subject_id="s1",
                blocks=(BlockData(block_id="b1", trials=(_trial(0, 1, 1.0), _trial(1, 1, 1.0))),),
            ),
            SubjectData(
                subject_id="s2",
                blocks=(BlockData(block_id="b1", trials=(_trial(0, 1, 1.0), _trial(1, 0, 0.0))),),
            ),
        )
    )
    return compare_study_candidate_models(
        study,
        candidate_specs=(
            CandidateFitSpec(
                name="good",
                fit_function=_constant_mle_fit(log_likelihood=-1.0, alpha=0.3),
                n_parameters=1,
            ),
            CandidateFitSpec(
                name="bad",
                fit_function=_constant_mle_fit(log_likelihood=-2.0, alpha=0.5),
                n_parameters=1,
            ),
        ),
        criterion="log_likelihood",
    )


def test_study_and_subject_comparison_record_helpers() -> None:
    """Serialization helpers should flatten study/subject comparison outputs."""

    result = _study_result()

    study_rows = study_model_comparison_records(result)
    assert len(study_rows) == 2
    assert {row["candidate_name"] for row in study_rows} == {"good", "bad"}

    subject_rows = study_model_comparison_subject_records(result)
    assert len(subject_rows) == 4
    assert {row["subject_id"] for row in subject_rows} == {"s1", "s2"}

    one_subject_rows = subject_model_comparison_records(result.subject_results[0])
    assert len(one_subject_rows) == 2
    assert one_subject_rows[0]["subject_id"] == "s1"


def test_study_and_subject_comparison_csv_writers(tmp_path: Path) -> None:
    """CSV writers should persist study aggregate and subject rows."""

    result = _study_result()
    study_path = write_study_model_comparison_csv(result, tmp_path / "study_cmp.csv")
    subjects_path = write_study_model_comparison_subject_csv(result, tmp_path / "study_cmp_subjects.csv")
    one_subject_path = write_subject_model_comparison_csv(
        result.subject_results[0],
        tmp_path / "subject_cmp.csv",
    )

    assert study_path.exists()
    assert subjects_path.exists()
    assert one_subject_path.exists()

    with study_path.open("r", encoding="utf-8", newline="") as handle:
        study_rows = list(csv.DictReader(handle))
    assert len(study_rows) == 2

    with subjects_path.open("r", encoding="utf-8", newline="") as handle:
        subject_rows = list(csv.DictReader(handle))
    assert len(subject_rows) == 4

    with one_subject_path.open("r", encoding="utf-8", newline="") as handle:
        one_subject_rows = list(csv.DictReader(handle))
    assert len(one_subject_rows) == 2
