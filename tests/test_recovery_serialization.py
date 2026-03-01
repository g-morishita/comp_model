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
                best_log_posterior=-1.5,
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
    assert rows[0]["best_log_posterior"] == pytest.approx(-1.5)

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
                        log_posterior=-1.2,
                        n_parameters=2,
                        aic=6.0,
                        bic=6.5,
                        waic=5.9,
                        psis_loo=6.1,
                        score=-1.0,
                        best_params={"alpha": 0.2},
                    ),
                    CandidateFitSummary(
                        candidate_name="cand2",
                        log_likelihood=-2.0,
                        log_posterior=None,
                        n_parameters=2,
                        aic=8.0,
                        bic=8.5,
                        waic=None,
                        psis_loo=None,
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
    row_by_name = {row["candidate_name"]: row for row in case_rows}
    assert row_by_name["cand1"]["log_posterior"] == pytest.approx(-1.2)
    assert row_by_name["cand2"]["log_posterior"] is None
    assert row_by_name["cand1"]["aic"] == pytest.approx(6.0)
    assert row_by_name["cand1"]["waic"] == pytest.approx(5.9)
    assert row_by_name["cand2"]["psis_loo"] is None
    assert row_by_name["cand1"]["param__alpha"] == pytest.approx(0.2)
    assert row_by_name["cand2"]["param__alpha"] == pytest.approx(0.8)

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
