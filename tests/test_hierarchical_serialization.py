"""Tests for hierarchical fitting serialization helpers."""

from __future__ import annotations

from comp_model.inference import (
    HierarchicalBlockResult,
    HierarchicalStudyMapResult,
    HierarchicalSubjectMapResult,
    hierarchical_study_block_records,
    hierarchical_study_summary_records,
    write_hierarchical_study_block_records_csv,
    write_hierarchical_study_summary_csv,
)
from comp_model.inference.mle import ScipyMinimizeDiagnostics


def _make_subject(subject_id: str, p1: float, p2: float) -> HierarchicalSubjectMapResult:
    """Build one synthetic hierarchical subject result row set."""

    return HierarchicalSubjectMapResult(
        subject_id=subject_id,
        parameter_names=("p_right",),
        group_location_z={"p_right": 0.0},
        group_scale_z={"p_right": 1.0},
        block_results=(
            HierarchicalBlockResult(
                block_id="b1",
                params={"p_right": p1},
                log_likelihood=-3.0,
            ),
            HierarchicalBlockResult(
                block_id="b2",
                params={"p_right": p2},
                log_likelihood=-2.0,
            ),
        ),
        total_log_likelihood=-5.0,
        total_log_prior=-1.0,
        total_log_posterior=-6.0,
        scipy_diagnostics=ScipyMinimizeDiagnostics(
            method="legacy",
            success=True,
            status=0,
            message="ok",
            n_iterations=1,
            n_function_evaluations=1,
        ),
    )


def _make_study_result() -> HierarchicalStudyMapResult:
    """Build one synthetic hierarchical study result."""

    return HierarchicalStudyMapResult(
        subject_results=(
            _make_subject("s1", 0.2, 0.7),
            _make_subject("s2", 0.3, 0.8),
        ),
        total_log_likelihood=-10.0,
        total_log_prior=-2.0,
        total_log_posterior=-12.0,
    )


def test_hierarchical_record_helpers_return_rows() -> None:
    """Record helpers should flatten hierarchical results into row dictionaries."""

    result = _make_study_result()
    block_rows = hierarchical_study_block_records(result)
    summary_rows = hierarchical_study_summary_records(result)

    assert len(block_rows) == 4
    assert len(summary_rows) == 2
    assert set(block_rows[0]) >= {"subject_id", "block_id", "log_likelihood", "param__p_right"}
    assert set(summary_rows[0]) >= {"subject_id", "total_log_likelihood", "total_log_prior", "total_log_posterior"}


def test_write_hierarchical_csv_helpers(tmp_path) -> None:
    """CSV writer helpers should persist hierarchical rows to disk."""

    result = _make_study_result()
    block_path = tmp_path / "hierarchical_blocks.csv"
    summary_path = tmp_path / "hierarchical_summary.csv"

    out_block = write_hierarchical_study_block_records_csv(result, block_path)
    out_summary = write_hierarchical_study_summary_csv(result, summary_path)

    assert out_block.exists()
    assert out_summary.exists()
    assert out_block.read_text(encoding="utf-8").startswith("subject_id,")
    assert out_summary.read_text(encoding="utf-8").startswith("subject_id,")

