"""Tests for unified best-fit result extraction helpers."""

from __future__ import annotations

import pytest

from comp_model.inference import (
    BayesFitResult,
    HierarchicalMCMCDraw,
    HierarchicalPosteriorCandidate,
    HierarchicalSubjectPosteriorResult,
    MCMCDiagnostics,
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


def test_extract_best_fit_summary_from_hierarchical_posterior_result() -> None:
    """Hierarchical posterior results should expose MAP block-mean parameters."""

    candidate = HierarchicalPosteriorCandidate(
        parameter_names=("alpha", "beta"),
        group_location_z={"alpha": 0.0, "beta": 0.0},
        group_scale_z={"alpha": 1.0, "beta": 1.0},
        block_params_z=(
            {"alpha": 0.1, "beta": 1.0},
            {"alpha": 0.2, "beta": 2.0},
        ),
        block_params=(
            {"alpha": 0.2, "beta": 2.0},
            {"alpha": 0.4, "beta": 4.0},
        ),
        log_likelihood=-12.0,
        log_prior=-1.5,
        log_posterior=-13.5,
    )
    result = HierarchicalSubjectPosteriorResult(
        subject_id="s1",
        parameter_names=("alpha", "beta"),
        draws=(HierarchicalMCMCDraw(candidate=candidate, accepted=True, iteration=0),),
        diagnostics=MCMCDiagnostics(
            method="within_subject_hierarchical_stan_nuts",
            n_iterations=10,
            n_warmup=2,
            n_kept_draws=1,
            thin=1,
            n_accepted=10,
            acceptance_rate=1.0,
            random_seed=None,
        ),
    )

    summary = extract_best_fit_summary(result)
    assert summary.params["alpha"] == pytest.approx(0.3)
    assert summary.params["beta"] == pytest.approx(3.0)
    assert summary.log_likelihood == pytest.approx(-12.0)
    assert summary.log_posterior == pytest.approx(-13.5)
