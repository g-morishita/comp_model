"""Tests for MAP fit serialization helpers."""

from __future__ import annotations

import csv
from pathlib import Path

from comp_model.inference.bayes import BayesFitResult, PosteriorCandidate
from comp_model.inference.map_study_fitting import MapBlockFitResult, MapStudyFitResult, MapSubjectFitResult
from comp_model.inference.serialization import (
    map_block_fit_records,
    map_study_fit_records,
    map_study_summary_records,
    map_subject_fit_records,
    write_map_study_fit_records_csv,
    write_map_study_fit_summary_csv,
)


def _mock_map_block(block_id: str, *, p_right: float, log_likelihood: float) -> MapBlockFitResult:
    """Build one mocked MAP block-fit result."""

    candidate = PosteriorCandidate(
        params={"p_right": p_right},
        log_likelihood=log_likelihood,
        log_prior=-0.2,
        log_posterior=log_likelihood - 0.2,
    )
    return MapBlockFitResult(
        block_id=block_id,
        n_trials=40,
        fit_result=BayesFitResult(
            map_candidate=candidate,
            candidates=(candidate,),
        ),
    )


def _mock_map_study_fit() -> MapStudyFitResult:
    """Build one mocked MAP study fit result."""

    s1 = MapSubjectFitResult(
        subject_id="s1",
        block_results=(_mock_map_block("b1", p_right=0.8, log_likelihood=-8.0),),
        total_log_likelihood=-8.0,
        total_log_posterior=-8.2,
        mean_map_params={"p_right": 0.8},
    )
    s2 = MapSubjectFitResult(
        subject_id="s2",
        block_results=(_mock_map_block("b2", p_right=0.6, log_likelihood=-9.0),),
        total_log_likelihood=-9.0,
        total_log_posterior=-9.2,
        mean_map_params={"p_right": 0.6},
    )
    return MapStudyFitResult(
        subject_results=(s1, s2),
        total_log_likelihood=-17.0,
        total_log_posterior=-17.4,
    )


def test_map_record_helpers_emit_expected_shapes() -> None:
    """MAP record builders should emit expected row counts and columns."""

    study = _mock_map_study_fit()
    block_rows = map_study_fit_records(study)
    summary_rows = map_study_summary_records(study)

    assert len(block_rows) == 2
    assert len(summary_rows) == 2
    assert set(block_rows[0]) >= {"subject_id", "block_id", "log_likelihood", "log_prior", "log_posterior"}
    assert set(summary_rows[0]) >= {"subject_id", "total_log_likelihood", "total_log_posterior"}

    subject_rows = map_subject_fit_records(study.subject_results[0])
    assert len(subject_rows) == 1
    block_rows_single = map_block_fit_records(study.subject_results[0].block_results[0], subject_id="s1")
    assert len(block_rows_single) == 1


def test_map_fit_serialization_csv_writers(tmp_path: Path) -> None:
    """MAP CSV writers should persist detail and summary rows."""

    study = _mock_map_study_fit()
    detail_path = write_map_study_fit_records_csv(study, tmp_path / "map_fit_rows.csv")
    summary_path = write_map_study_fit_summary_csv(study, tmp_path / "map_fit_summary.csv")

    assert detail_path.exists()
    assert summary_path.exists()

    with detail_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
