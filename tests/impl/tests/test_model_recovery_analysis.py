"""Tests for model recovery analysis helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from comp_model_impl.analysis.model_selection import add_information_criteria
from comp_model_impl.recovery.model.analysis import (
    confusion_matrix,
    summarize_delta_scores,
)


def test_confusion_matrix_happy_path() -> None:
    """Confusion matrix should count selected models by generator."""
    winners = pd.DataFrame(
        {
            "generating_model": ["A", "A", "B"],
            "selected_model": ["A", "B", "B"],
        }
    )
    cm = confusion_matrix(winners)

    assert cm.loc["A", "A"] == 1
    assert cm.loc["A", "B"] == 1
    assert cm.loc["B", "B"] == 1


def test_confusion_matrix_empty_and_missing_columns() -> None:
    """Confusion matrix should handle empty tables and reject malformed ones."""
    out = confusion_matrix(pd.DataFrame())
    assert out.empty

    with pytest.raises(ValueError, match="generating_model and selected_model"):
        _ = confusion_matrix(pd.DataFrame({"generating_model": ["A"]}))


def test_summarize_delta_scores_happy_path() -> None:
    """Delta-score summaries should include aggregate and quantile columns."""
    winners = pd.DataFrame(
        {
            "generating_model": ["A", "A", "B", "B"],
            "delta_to_second": [1.0, 3.0, 2.0, 4.0],
        }
    )
    out = summarize_delta_scores(winners)
    by_gen = {row["generating_model"]: row for _, row in out.iterrows()}

    assert by_gen["A"]["delta_mean"] == pytest.approx(2.0)
    assert by_gen["A"]["delta_median"] == pytest.approx(2.0)
    assert by_gen["B"]["delta_mean"] == pytest.approx(3.0)
    assert by_gen["B"]["delta_median"] == pytest.approx(3.0)


def test_summarize_delta_scores_empty_or_missing_column() -> None:
    """Delta-score summaries should return an empty typed table when unavailable."""
    out_empty = summarize_delta_scores(pd.DataFrame())
    assert list(out_empty.columns) == [
        "generating_model",
        "delta_mean",
        "delta_median",
        "delta_p10",
        "delta_p90",
    ]
    assert out_empty.empty

    out_missing = summarize_delta_scores(pd.DataFrame({"generating_model": ["A"]}))
    assert out_missing.empty


def test_add_information_criteria_single_group() -> None:
    """AIC/BIC metrics and BIC-BF should match closed-form values."""
    fit = pd.DataFrame(
        {
            "candidate_model": ["m1", "m2"],
            "ll_total": [-100.0, -110.0],
            "k_total": [5, 5],
            "n_obs_total": [200, 200],
        }
    )
    out = add_information_criteria(fit, group_cols=())

    m1 = out.loc[out["candidate_model"] == "m1"].iloc[0]
    m2 = out.loc[out["candidate_model"] == "m2"].iloc[0]

    assert m1["aic"] == pytest.approx(210.0)
    assert m2["aic"] == pytest.approx(230.0)
    assert m1["delta_bic"] == pytest.approx(0.0)
    assert m2["delta_bic"] == pytest.approx(20.0)
    assert m1["bf_best_vs_model_bic"] == pytest.approx(1.0)
    assert m2["bf_best_vs_model_bic"] == pytest.approx(float(np.exp(10.0)))


def test_add_information_criteria_grouped_weights_sum_to_one() -> None:
    """Akaike/BIC weights should normalize within each comparison group."""
    fit = pd.DataFrame(
        {
            "rep": [0, 0, 1, 1],
            "generating_model": ["A", "A", "B", "B"],
            "candidate_model": ["m1", "m2", "m1", "m2"],
            "ll_total": [-50.0, -52.0, -80.0, -82.0],
            "k_total": [3, 3, 3, 3],
            "n_obs_total": [120, 120, 120, 120],
        }
    )
    out = add_information_criteria(fit)

    for _, grp in out.groupby(["rep", "generating_model"]):
        assert float(grp["akaike_weight"].sum()) == pytest.approx(1.0)
        assert float(grp["bic_weight"].sum()) == pytest.approx(1.0)


def test_add_information_criteria_requires_columns() -> None:
    """Missing required columns should raise a clear error."""
    with pytest.raises(ValueError, match="missing required column"):
        _ = add_information_criteria(pd.DataFrame({"ll_total": [-1.0]}))


def test_add_information_criteria_waic_from_waic_column() -> None:
    """WAIC deltas and weights should be computed when WAIC is present."""
    fit = pd.DataFrame(
        {
            "candidate_model": ["m1", "m2"],
            "ll_total": [-100.0, -110.0],
            "k_total": [5, 5],
            "n_obs_total": [200, 200],
            "waic": [210.0, 220.0],
        }
    )
    out = add_information_criteria(fit, group_cols=())

    m1 = out.loc[out["candidate_model"] == "m1"].iloc[0]
    m2 = out.loc[out["candidate_model"] == "m2"].iloc[0]

    assert m1["delta_waic"] == pytest.approx(0.0)
    assert m2["delta_waic"] == pytest.approx(10.0)
    assert m1["waic_weight"] + m2["waic_weight"] == pytest.approx(1.0)
    assert m1["waic_weight"] > m2["waic_weight"]


def test_add_information_criteria_waic_from_elpd_waic_column() -> None:
    """WAIC should be derived as -2 * elpd_waic when only ELPD is available."""
    fit = pd.DataFrame(
        {
            "candidate_model": ["m1", "m2"],
            "ll_total": [-100.0, -110.0],
            "k_total": [5, 5],
            "n_obs_total": [200, 200],
            "elpd_waic": [-105.0, -110.0],
        }
    )
    out = add_information_criteria(fit, group_cols=())

    m1 = out.loc[out["candidate_model"] == "m1"].iloc[0]
    m2 = out.loc[out["candidate_model"] == "m2"].iloc[0]

    assert m1["waic"] == pytest.approx(210.0)
    assert m2["waic"] == pytest.approx(220.0)
    assert m1["delta_waic"] == pytest.approx(0.0)
    assert m2["delta_waic"] == pytest.approx(10.0)


def test_add_information_criteria_waic_missing_declared_column() -> None:
    """Declaring a WAIC column that is absent should raise a clear error."""
    fit = pd.DataFrame(
        {
            "candidate_model": ["m1"],
            "ll_total": [-100.0],
            "k_total": [5],
            "n_obs_total": [200],
        }
    )
    with pytest.raises(ValueError, match="missing required WAIC column"):
        _ = add_information_criteria(fit, waic_col="waic_custom")
