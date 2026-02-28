"""Tests for recovery result serialization helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from comp_model.inference import MLECandidate
from comp_model.recovery.model import CandidateFitSummary, ModelRecoveryCase, ModelRecoveryResult
from comp_model.recovery.parameter import ParameterRecoveryCase, ParameterRecoveryResult
from comp_model.recovery.serialization import (
    model_recovery_case_records,
    model_recovery_confusion_records,
    parameter_recovery_records,
    write_model_recovery_cases_csv,
    write_model_recovery_confusion_csv,
    write_parameter_recovery_csv,
    write_records_csv,
)


def test_parameter_recovery_records_and_csv_roundtrip(tmp_path: Path) -> None:
    """Parameter recovery serialization should emit stable row structure and CSV."""

    result = ParameterRecoveryResult(
        cases=(
            ParameterRecoveryCase(
                case_index=0,
                simulation_seed=1,
                true_params={"alpha": 0.2},
                estimated_params={"alpha": 0.3},
                best_log_likelihood=-1.23,
            ),
        ),
        mean_absolute_error={"alpha": 0.1},
        mean_signed_error={"alpha": 0.1},
    )

    rows = parameter_recovery_records(result)
    assert len(rows) == 1
    assert rows[0]["true__alpha"] == pytest.approx(0.2)
    assert rows[0]["estimated__alpha"] == pytest.approx(0.3)
    assert rows[0]["error__alpha"] == pytest.approx(0.1)

    output = write_parameter_recovery_csv(result, tmp_path / "parameter_recovery.csv")
    assert output.exists()

    with output.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        data = list(reader)

    assert len(data) == 1
    assert float(data[0]["true__alpha"]) == pytest.approx(0.2)


def test_model_recovery_records_and_csv_roundtrip(tmp_path: Path) -> None:
    """Model recovery serialization should emit case rows and confusion rows."""

    result = ModelRecoveryResult(
        cases=(
            ModelRecoveryCase(
                case_index=0,
                generating_model_name="genA",
                simulation_seed=3,
                selected_candidate_name="cand1",
                candidate_summaries=(
                    CandidateFitSummary(
                        candidate_name="cand1",
                        log_likelihood=-1.0,
                        n_parameters=2,
                        score=-1.0,
                        best_params={"alpha": 0.2},
                    ),
                    CandidateFitSummary(
                        candidate_name="cand2",
                        log_likelihood=-2.0,
                        n_parameters=2,
                        score=-2.0,
                        best_params={"alpha": 0.8},
                    ),
                ),
            ),
        ),
        confusion_matrix={"genA": {"cand1": 1}},
        criterion="log_likelihood",
    )

    case_rows = model_recovery_case_records(result)
    assert len(case_rows) == 2
    assert {row["candidate_name"] for row in case_rows} == {"cand1", "cand2"}

    confusion_rows = model_recovery_confusion_records(result)
    assert confusion_rows == [
        {
            "generating_model_name": "genA",
            "selected_candidate_name": "cand1",
            "count": 1,
            "criterion": "log_likelihood",
        }
    ]

    cases_csv = write_model_recovery_cases_csv(result, tmp_path / "model_recovery_cases.csv")
    confusion_csv = write_model_recovery_confusion_csv(result, tmp_path / "model_recovery_confusion.csv")
    assert cases_csv.exists()
    assert confusion_csv.exists()


def test_write_records_csv_rejects_empty_input(tmp_path: Path) -> None:
    """CSV writer should reject empty row collections."""

    with pytest.raises(ValueError, match="rows must not be empty"):
        write_records_csv([], tmp_path / "empty.csv")
