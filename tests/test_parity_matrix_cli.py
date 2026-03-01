"""Tests for parity matrix CLI helper."""

from __future__ import annotations

import json

import pytest

from comp_model.analysis import run_model_parity_matrix_cli


def test_parity_matrix_cli_writes_json_and_csv(tmp_path, capsys) -> None:
    """CLI should write requested matrix artifacts and return success code."""

    json_path = tmp_path / "parity_matrix.json"
    csv_path = tmp_path / "parity_matrix.csv"

    code = run_model_parity_matrix_cli(
        [
            "--output-json",
            str(json_path),
            "--output-csv",
            str(csv_path),
        ]
    )
    captured = capsys.readouterr()

    assert code == 0
    assert json_path.exists()
    assert csv_path.exists()
    assert "Parity matrix summary" in captured.out

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["n_invalid"] == 0


def test_parity_matrix_cli_requires_at_least_one_output() -> None:
    """CLI should reject invocation without output targets."""

    with pytest.raises(ValueError, match="at least one of --output-json or --output-csv"):
        run_model_parity_matrix_cli([])
