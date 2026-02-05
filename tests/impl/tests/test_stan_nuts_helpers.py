"""Unit tests for Stan NUTS helper utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from comp_model_impl.estimators.stan.nuts import (
    _delta_labels,
    _flatten_mean,
    _is_within_subject_model,
    _load_yaml,
    _safe_summary_metric,
    _strip_hat,
)
from comp_model_impl.models.qrl.qrl import QRL
from comp_model_impl.models.within_subject_shared_delta import ConditionedSharedDeltaModel


@dataclass
class _DummySeries:
    """Minimal series-like object with max/min."""

    values: list[float]

    def max(self) -> float:
        """Return the max of the series."""
        return max(self.values)

    def min(self) -> float:
        """Return the min of the series."""
        return min(self.values)


@dataclass
class _DummySummary:
    """Minimal dataframe-like object for summary metrics."""

    columns: list[str]
    data: dict[str, _DummySeries]

    def __getitem__(self, key: str) -> _DummySeries:
        """Return the series for a column."""
        return self.data[key]


def test_strip_hat():
    """_strip_hat removes trailing suffixes and leaves other names unchanged."""
    assert _strip_hat("alpha_hat") == "alpha"
    assert _strip_hat("beta") == "beta"


def test_delta_labels_excludes_baseline():
    """_delta_labels returns non-baseline labels in order."""
    labels = ["A", "B", "C"]
    assert _delta_labels(labels, baseline_idx_1based=2) == ["A", "C"]


def test_flatten_mean_scalar_vector_matrix_and_tensor():
    """_flatten_mean flattens scalar, vector, matrix, and tensor inputs."""
    assert _flatten_mean(name="alpha_hat", mean=1.2) == {"alpha": 1.2}

    out_vec = _flatten_mean(name="beta_hat", mean=[0.1, 0.2], condition_labels=["A", "B"])
    assert out_vec == {"beta__A": 0.1, "beta__B": 0.2}

    out_delta = _flatten_mean(
        name="gamma__delta_hat",
        mean=[-0.5, 0.25],
        condition_labels=["A", "B", "C"],
        baseline_idx_1based=2,
    )
    assert out_delta == {"gamma__delta__A": -0.5, "gamma__delta__C": 0.25}

    out_mat = _flatten_mean(name="mu_hat", mean=[[1.0, 2.0], [3.0, 4.0]])
    assert out_mat == {
        "mu[1,1]": 1.0,
        "mu[1,2]": 2.0,
        "mu[2,1]": 3.0,
        "mu[2,2]": 4.0,
    }

    out_tens = _flatten_mean(name="tau_hat", mean=[[[1.0, 2.0], [3.0, 4.0]]])
    assert out_tens == {
        "tau[1,1,1]": 1.0,
        "tau[1,1,2]": 2.0,
        "tau[1,2,1]": 3.0,
        "tau[1,2,2]": 4.0,
    }


def test_safe_summary_metric_handles_missing_and_agg():
    """_safe_summary_metric extracts max/min and handles missing columns."""
    summary = _DummySummary(
        columns=["R_hat", "ESS_bulk"],
        data={
            "R_hat": _DummySeries([1.1, 1.2]),
            "ESS_bulk": _DummySeries([200, 150]),
        },
    )
    assert _safe_summary_metric(summary, ["R_hat"], agg="max") == 1.2
    assert _safe_summary_metric(summary, ["ESS_bulk"], agg="min") == 150
    assert _safe_summary_metric(summary, ["missing"], agg="max") is None


def test_is_within_subject_model_flag():
    """_is_within_subject_model identifies shared+delta wrappers."""
    base = QRL()
    ws = ConditionedSharedDeltaModel(base_model=base, conditions=["A", "B"], baseline_condition="A")
    assert _is_within_subject_model(base) is False
    assert _is_within_subject_model(ws) is True


def test_load_yaml_reads_mapping(tmp_path):
    """_load_yaml reads a YAML mapping from disk."""
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "cfg.yaml"
    path.write_text("a: 1\nb: 2\n", encoding="utf-8")
    cfg = _load_yaml(str(path))
    assert isinstance(cfg, dict)
    assert cfg == {"a": 1, "b": 2}
