"""Tests for parameter recovery analysis metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from comp_model_impl.recovery.parameter.analysis import (
    compute_parameter_recovery_metrics,
    compute_population_recovery_metrics,
)


def test_compute_parameter_recovery_metrics_values() -> None:
    """Metrics should match simple hand-computed values."""
    df = pd.DataFrame(
        {
            "rep": [0, 0, 0],
            "subject_id": ["s1", "s2", "s3"],
            "param": ["alpha", "alpha", "alpha"],
            "true": [0.0, 1.0, 2.0],
            "hat": [0.0, 2.0, 1.0],
        }
    )

    metrics = compute_parameter_recovery_metrics(df)
    assert list(metrics["param"]) == ["alpha"]
    assert list(metrics["rep"]) == [0]
    row = metrics.iloc[0]

    err = np.array([0.0, 1.0, -1.0], dtype=float)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    med_abs = float(np.median(np.abs(err)))
    bias = float(np.mean(err))

    assert row["n"] == 3
    assert row["rmse"] == pytest.approx(rmse)
    assert row["mae"] == pytest.approx(mae)
    assert row["median_abs_error"] == pytest.approx(med_abs)
    assert row["bias"] == pytest.approx(bias)


def test_compute_population_recovery_metrics_pools_reps() -> None:
    """Population metrics should pool across reps (one estimate per rep)."""
    df = pd.DataFrame(
        {
            "rep": [0, 1, 2],
            "subject_id": ["POP", "POP", "POP"],
            "param": ["alpha", "alpha", "alpha"],
            "true": [0.0, 1.0, 2.0],
            "hat": [0.0, 2.0, 1.0],
        }
    )
    metrics = compute_population_recovery_metrics(df)
    assert list(metrics["param"]) == ["alpha"]
    row = metrics.iloc[0]

    err = np.array([0.0, 1.0, -1.0], dtype=float)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    med_abs = float(np.median(np.abs(err)))
    bias = float(np.mean(err))
    corr = float(np.corrcoef(df["true"].to_numpy(), df["hat"].to_numpy())[0, 1])

    assert row["rep"] == -1
    assert row["n"] == 3
    assert row["rmse"] == pytest.approx(rmse)
    assert row["mae"] == pytest.approx(mae)
    assert row["median_abs_error"] == pytest.approx(med_abs)
    assert row["bias"] == pytest.approx(bias)
    assert row["corr"] == pytest.approx(corr)


def test_compute_parameter_recovery_metrics_splits_by_rep() -> None:
    """Metrics should be computed per replication."""
    df = pd.DataFrame(
        {
            "rep": [0, 0, 1, 1],
            "subject_id": ["s1", "s2", "s1", "s2"],
            "param": ["alpha", "alpha", "alpha", "alpha"],
            "true": [0.0, 1.0, 0.0, 1.0],
            "hat": [0.0, 1.0, 1.0, 0.0],
        }
    )
    metrics = compute_parameter_recovery_metrics(df)
    assert list(metrics["rep"]) == [0, 1]
    assert list(metrics["param"]) == ["alpha", "alpha"]


def test_compute_parameter_recovery_metrics_requires_columns() -> None:
    """Missing required columns should raise a ValueError."""
    df = pd.DataFrame({"param": ["alpha"], "true": [0.1]})
    with pytest.raises(ValueError):
        _ = compute_parameter_recovery_metrics(df)
