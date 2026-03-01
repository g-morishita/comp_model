"""Tests for model-comparison serialization helpers."""

from __future__ import annotations

import csv
from pathlib import Path

from comp_model.inference import BayesFitResult, MLECandidate, MLEFitResult, PosteriorCandidate
from comp_model.inference.model_selection import CandidateComparison, ModelComparisonResult
from comp_model.inference.serialization import model_comparison_records, write_model_comparison_csv


def _mock_result() -> ModelComparisonResult:
    """Build one mixed MLE/MAP model-comparison result."""

    mle_fit = MLEFitResult(
        best=MLECandidate(params={"alpha": 0.3}, log_likelihood=-10.0),
        candidates=(MLECandidate(params={"alpha": 0.3}, log_likelihood=-10.0),),
    )
    map_candidate = PosteriorCandidate(
        params={"alpha": 0.4},
        log_likelihood=-9.0,
        log_prior=-0.5,
        log_posterior=-9.5,
    )
    map_fit = BayesFitResult(
        map_candidate=map_candidate,
        candidates=(map_candidate,),
    )

    return ModelComparisonResult(
        criterion="log_likelihood",
        n_observations=100,
        selected_candidate_name="map_candidate",
        comparisons=(
            CandidateComparison(
                candidate_name="mle_candidate",
                log_likelihood=-10.0,
                n_parameters=1,
                aic=22.0,
                bic=24.6,
                score=-10.0,
                fit_result=mle_fit,
            ),
            CandidateComparison(
                candidate_name="map_candidate",
                log_likelihood=-9.0,
                n_parameters=1,
                aic=20.0,
                bic=22.6,
                score=-9.0,
                fit_result=map_fit,
            ),
        ),
    )


def test_model_comparison_records_include_selected_and_params() -> None:
    """Model-comparison rows should include candidate metrics and best params."""

    rows = model_comparison_records(_mock_result())
    assert len(rows) == 2

    row_by_name = {row["candidate_name"]: row for row in rows}
    assert row_by_name["mle_candidate"]["is_selected"] is False
    assert row_by_name["map_candidate"]["is_selected"] is True
    assert row_by_name["mle_candidate"]["param__alpha"] == 0.3
    assert row_by_name["map_candidate"]["param__alpha"] == 0.4
    assert row_by_name["mle_candidate"]["log_posterior"] is None
    assert row_by_name["map_candidate"]["log_posterior"] == -9.5


def test_write_model_comparison_csv(tmp_path: Path) -> None:
    """Model-comparison CSV writer should persist candidate rows."""

    path = write_model_comparison_csv(_mock_result(), tmp_path / "model_comparison.csv")
    assert path.exists()

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["candidate_name"] for row in rows} == {"mle_candidate", "map_candidate"}
