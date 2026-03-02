"""Tests for fit-result serialization helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from comp_model.inference.mle import MLECandidate, MLEFitResult
from comp_model.inference.serialization import (
    block_fit_records,
    study_fit_records,
    study_summary_records,
    subject_fit_records,
    write_records_csv,
    write_study_fit_records_csv,
    write_study_fit_summary_csv,
)
from comp_model.inference.study_fitting import BlockFitResult, StudyFitResult, SubjectFitResult


def _mock_block(block_id: str) -> BlockFitResult:
    """Build one mocked block fit result for serialization tests."""

    c1 = MLECandidate(params={"alpha": 0.2, "beta": 2.0}, log_likelihood=-10.0)
    c2 = MLECandidate(params={"alpha": 0.5, "beta": 2.0}, log_likelihood=-8.0)
    return BlockFitResult(
        block_id=block_id,
        n_trials=40,
        fit_result=MLEFitResult(best=c2, candidates=(c1, c2)),
    )


def _mock_study_fit() -> StudyFitResult:
    """Build one mocked study-fit result."""

    s1 = SubjectFitResult(
        subject_id="s1",
        block_results=(_mock_block("b1"),),
        total_log_likelihood=-8.0,
        mean_best_params={"alpha": 0.5, "beta": 2.0},
    )
    s2 = SubjectFitResult(
        subject_id="s2",
        block_results=(_mock_block("b2"),),
        total_log_likelihood=-8.0,
        mean_best_params={"alpha": 0.5, "beta": 2.0},
    )
    return StudyFitResult(subject_results=(s1, s2), total_log_likelihood=-16.0)


def test_block_and_subject_records_shapes() -> None:
    """Block/subject record builders should emit expected row counts."""

    block = _mock_block("b1")
    block_rows = block_fit_records(block, subject_id="s1")
    assert len(block_rows) == 2
    assert any(row["is_best"] for row in block_rows)

    subject = SubjectFitResult(
        subject_id="s1",
        block_results=(block,),
        total_log_likelihood=-8.0,
        mean_best_params={"alpha": 0.5, "beta": 2.0},
    )
    subject_rows = subject_fit_records(subject)
    assert len(subject_rows) == 2


def test_study_records_and_summary_rows() -> None:
    """Study record builders should emit block-level and summary-level rows."""

    study = _mock_study_fit()
    candidate_rows = study_fit_records(study)
    summary_rows = study_summary_records(study)

    assert len(candidate_rows) == 4
    assert len(summary_rows) == 2
    assert {row["subject_id"] for row in summary_rows} == {"s1", "s2"}
    assert {row["fit_mode"] for row in summary_rows} == {"independent"}
    assert {row["input_n_blocks"] for row in summary_rows} == {1}


def test_fit_serialization_csv_writers(tmp_path: Path) -> None:
    """CSV writers should persist candidate rows and summary rows."""

    study = _mock_study_fit()

    detail_path = write_study_fit_records_csv(study, tmp_path / "fit_rows.csv")
    summary_path = write_study_fit_summary_csv(study, tmp_path / "fit_summary.csv")

    assert detail_path.exists()
    assert summary_path.exists()

    with detail_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2


def test_write_records_csv_rejects_empty_rows(tmp_path: Path) -> None:
    """Generic writer should reject empty record lists."""

    with pytest.raises(ValueError, match="rows must not be empty"):
        write_records_csv([], tmp_path / "empty.csv")
