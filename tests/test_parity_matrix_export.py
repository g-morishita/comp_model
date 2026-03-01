"""Tests for machine-readable model parity matrix exports."""

from __future__ import annotations

import json

from comp_model.analysis import (
    build_model_parity_matrix,
    summarize_model_parity_matrix,
    write_model_parity_matrix_csv,
    write_model_parity_matrix_json,
)
from comp_model.models import V1_MODEL_PARITY


def test_build_model_parity_matrix_has_one_row_per_legacy_entry() -> None:
    """Parity matrix should preserve parity table cardinality."""

    rows = build_model_parity_matrix()
    assert len(rows) == len(V1_MODEL_PARITY)
    assert {row.legacy_name for row in rows} == {
        entry.legacy_name for entry in V1_MODEL_PARITY
    }


def test_model_parity_matrix_rows_for_implemented_entries_are_valid() -> None:
    """Implemented entries should resolve to valid class/registry targets."""

    rows = build_model_parity_matrix()
    implemented = [row for row in rows if row.status == "implemented"]

    assert implemented
    assert all(row.class_exists for row in implemented)
    assert all(row.mapping_valid for row in implemented)
    assert all(
        row.component_registered in (None, True)
        for row in implemented
    )


def test_model_parity_matrix_summary_and_serialization(tmp_path) -> None:
    """Summary and JSON/CSV outputs should be generated and readable."""

    rows = build_model_parity_matrix()
    summary = summarize_model_parity_matrix(rows)

    assert summary.n_rows == len(rows)
    assert summary.n_implemented + summary.n_planned == summary.n_rows
    assert summary.n_invalid == 0

    json_path = write_model_parity_matrix_json(rows, tmp_path / "parity_matrix.json")
    csv_path = write_model_parity_matrix_csv(rows, tmp_path / "parity_matrix.csv")

    assert json_path.exists()
    assert csv_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["n_invalid"] == 0
    assert len(payload["rows"]) == len(rows)
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "legacy_name" in csv_text
    assert "mapping_valid" in csv_text
