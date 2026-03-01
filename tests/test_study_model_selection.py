"""Tests for subject/study model-comparison helpers."""

from __future__ import annotations

from comp_model.core.data import BlockData, StudyData, SubjectData, TrialDecision
from comp_model.inference import (
    BayesFitResult,
    CandidateFitSpec,
    MLECandidate,
    MLEFitResult,
    PosteriorCandidate,
    compare_study_candidate_models,
    compare_subject_candidate_models,
)


def _trial(trial_index: int, action: int, reward: float) -> TrialDecision:
    """Build one trial row for study-comparison tests."""

    return TrialDecision(
        trial_index=trial_index,
        decision_index=0,
        actor_id="subject",
        available_actions=(0, 1),
        action=action,
        observation={"state": 0},
        outcome={"reward": reward},
    )


def _constant_mle_fit(*, log_likelihood: float, alpha: float = 0.3):
    """Build deterministic MLE fit function."""

    def _fit(trace):
        candidate = MLECandidate(params={"alpha": alpha}, log_likelihood=log_likelihood)
        return MLEFitResult(best=candidate, candidates=(candidate,))

    return _fit


def _constant_map_fit(*, log_likelihood: float, log_prior: float, alpha: float = 0.4):
    """Build deterministic MAP fit function."""

    def _fit(trace):
        candidate = PosteriorCandidate(
            params={"alpha": alpha},
            log_likelihood=log_likelihood,
            log_prior=log_prior,
            log_posterior=log_likelihood + log_prior,
        )
        return BayesFitResult(map_candidate=candidate, candidates=(candidate,))

    return _fit


def _subject(subject_id: str) -> SubjectData:
    """Build one toy subject with two blocks."""

    return SubjectData(
        subject_id=subject_id,
        blocks=(
            BlockData(block_id="b1", trials=(_trial(0, 1, 1.0), _trial(1, 1, 1.0))),
            BlockData(block_id="b2", trials=(_trial(0, 1, 1.0), _trial(1, 0, 0.0))),
        ),
    )


def test_compare_subject_candidate_models_aggregates_blocks() -> None:
    """Subject-level comparison should aggregate likelihoods across blocks."""

    subject = _subject("s1")
    result = compare_subject_candidate_models(
        subject,
        candidate_specs=(
            CandidateFitSpec(
                name="mle_bad",
                fit_function=_constant_mle_fit(log_likelihood=-10.0),
                n_parameters=1,
            ),
            CandidateFitSpec(
                name="map_good",
                fit_function=_constant_map_fit(log_likelihood=-8.0, log_prior=-0.5),
                n_parameters=1,
            ),
        ),
        criterion="log_likelihood",
    )

    assert result.subject_id == "s1"
    assert result.n_observations == 4
    assert result.selected_candidate_name == "map_good"
    by_name = {row.candidate_name: row for row in result.comparisons}
    assert by_name["mle_bad"].log_likelihood == -20.0
    assert by_name["mle_bad"].log_posterior is None
    assert by_name["map_good"].log_likelihood == -16.0
    assert by_name["map_good"].log_posterior == -17.0


def test_compare_study_candidate_models_aggregates_subjects() -> None:
    """Study-level comparison should aggregate subject-level results."""

    study = StudyData(subjects=(_subject("s1"), _subject("s2")))
    result = compare_study_candidate_models(
        study,
        candidate_specs=(
            CandidateFitSpec(
                name="mle_bad",
                fit_function=_constant_mle_fit(log_likelihood=-9.0),
                n_parameters=1,
            ),
            CandidateFitSpec(
                name="map_good",
                fit_function=_constant_map_fit(log_likelihood=-7.0, log_prior=-0.5),
                n_parameters=1,
            ),
        ),
        criterion="log_likelihood",
    )

    assert result.n_subjects == 2
    assert result.n_observations == 8
    assert result.selected_candidate_name == "map_good"
    by_name = {row.candidate_name: row for row in result.comparisons}
    assert by_name["mle_bad"].log_likelihood == -36.0
    assert by_name["map_good"].log_likelihood == -28.0
