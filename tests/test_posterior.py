"""Tests for posterior sample containers and summaries."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from comp_model.inference import (
    PosteriorSamples,
    posterior_summary_records,
    summarize_posterior,
    write_posterior_summary_csv,
)


def test_posterior_samples_basic_stats() -> None:
    """PosteriorSamples should expose draw count and moments."""

    samples = PosteriorSamples(
        parameter_draws={
            "alpha": np.asarray([0.1, 0.2, 0.3], dtype=float),
            "beta": np.asarray([1.0, 2.0, 3.0], dtype=float),
        }
    )
    assert samples.n_draws == 3
    assert samples.parameter_names == ("alpha", "beta")
    assert samples.mean("alpha") == pytest.approx(0.2)
    assert samples.std("beta") == pytest.approx(1.0)
    assert samples.quantile("alpha", 0.5) == pytest.approx(0.2)


def test_summarize_posterior_and_records() -> None:
    """Posterior summary utilities should produce stable rows."""

    samples = PosteriorSamples(
        parameter_draws={
            "alpha": np.asarray([0.1, 0.2, 0.3, 0.4], dtype=float),
        }
    )
    summary = summarize_posterior(samples, quantiles=(0.25, 0.5, 0.75))
    assert summary.n_draws == 4
    by_name = summary.by_name()
    assert by_name["alpha"].mean == pytest.approx(0.25)
    assert by_name["alpha"].quantiles[0.5] == pytest.approx(0.25)

    rows = posterior_summary_records(summary)
    assert len(rows) == 1
    assert rows[0]["parameter_name"] == "alpha"
    assert rows[0]["q0.500"] == pytest.approx(0.25)


def test_posterior_samples_validate_shapes() -> None:
    """PosteriorSamples should reject empty or inconsistent draws."""

    with pytest.raises(ValueError, match="must not be empty"):
        PosteriorSamples(parameter_draws={})

    with pytest.raises(ValueError, match="equal length"):
        PosteriorSamples(
            parameter_draws={
                "alpha": np.asarray([0.1, 0.2], dtype=float),
                "beta": np.asarray([1.0], dtype=float),
            }
        )

    with pytest.raises(ValueError, match="must be 1D"):
        PosteriorSamples(
            parameter_draws={
                "alpha": np.asarray([[0.1, 0.2]], dtype=float),
            }
        )


def test_write_posterior_summary_csv(tmp_path: Path) -> None:
    """Posterior summary CSV writer should persist one row per parameter."""

    samples = PosteriorSamples(
        parameter_draws={
            "alpha": np.asarray([0.1, 0.2, 0.3], dtype=float),
            "beta": np.asarray([1.0, 2.0, 3.0], dtype=float),
        }
    )
    summary = summarize_posterior(samples, quantiles=(0.5,))
    output = write_posterior_summary_csv(summary, tmp_path / "posterior_summary.csv")

    assert output.exists()
    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert {row["parameter_name"] for row in rows} == {"alpha", "beta"}
