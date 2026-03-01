"""Tests for unified best-fit result extraction helpers."""

from __future__ import annotations

import pytest

from comp_model.inference import (
    BayesFitResult,
    MLECandidate,
    MLEFitResult,
    PosteriorCandidate,
    extract_best_fit_summary,
)


def test_extract_best_fit_summary_from_mle_result() -> None:
    """MLE-style results should expose best params and log-likelihood."""

    result = MLEFitResult(
        best=MLECandidate(params={"alpha": 0.3}, log_likelihood=-10.0),
        candidates=(MLECandidate(params={"alpha": 0.3}, log_likelihood=-10.0),),
    )
    summary = extract_best_fit_summary(result)

    assert summary.params == {"alpha": 0.3}
    assert summary.log_likelihood == pytest.approx(-10.0)
    assert summary.log_posterior is None


def test_extract_best_fit_summary_from_map_result() -> None:
    """MAP-style results should expose both likelihood and posterior values."""

    result = BayesFitResult(
        map_candidate=PosteriorCandidate(
            params={"alpha": 0.4},
            log_likelihood=-9.0,
            log_prior=-0.5,
            log_posterior=-9.5,
        ),
        candidates=(),
    )
    summary = extract_best_fit_summary(result)

    assert summary.params == {"alpha": 0.4}
    assert summary.log_likelihood == pytest.approx(-9.0)
    assert summary.log_posterior == pytest.approx(-9.5)


def test_extract_best_fit_summary_rejects_unsupported_shape() -> None:
    """Unsupported fit-result objects should fail with a clear error."""

    with pytest.raises(TypeError, match="unsupported fit result type"):
        extract_best_fit_summary(object())
