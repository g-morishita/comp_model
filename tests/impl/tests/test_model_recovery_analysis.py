"""Tests for model recovery analysis helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from comp_model_impl.recovery.model.analysis import (
    confusion_matrix,
    recovery_rates,
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


def test_recovery_rates_happy_path() -> None:
    """Recovery rates should be correct per generating model."""
    winners = pd.DataFrame(
        {
            "generating_model": ["A", "A", "B", "B", "B"],
            "selected_model": ["A", "B", "B", "A", "B"],
        }
    )
    rates = recovery_rates(winners)
    by_gen = {row["generating_model"]: row for _, row in rates.iterrows()}

    assert by_gen["A"]["n"] == 2
    assert by_gen["A"]["n_correct"] == 1
    assert by_gen["A"]["recovery_rate"] == pytest.approx(0.5)

    assert by_gen["B"]["n"] == 3
    assert by_gen["B"]["n_correct"] == 2
    assert by_gen["B"]["recovery_rate"] == pytest.approx(2.0 / 3.0)


def test_recovery_rates_empty() -> None:
    """Recovery rates should return an empty typed table for empty input."""
    out = recovery_rates(pd.DataFrame())
    assert list(out.columns) == ["generating_model", "n", "n_correct", "recovery_rate"]
    assert out.empty


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
