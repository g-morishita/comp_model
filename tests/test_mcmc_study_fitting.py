"""Tests for study-level MCMC posterior sampling workflows."""

from __future__ import annotations

import math

from comp_model.core.data import BlockData, StudyData, SubjectData, TrialDecision
from comp_model.inference import (
    IndependentPriorProgram,
    sample_posterior_block_data,
    sample_posterior_study_data,
    sample_posterior_subject_data,
    uniform_log_prior,
)


def _trial(trial_index: int, action: int, reward: float) -> TrialDecision:
    """Build one trial row for MCMC study-fitting tests."""

    return TrialDecision(
        trial_index=trial_index,
        decision_index=0,
        actor_id="subject",
        available_actions=(0, 1),
        action=action,
        observation={"state": 0},
        outcome={"reward": reward},
    )


def _prior_program() -> IndependentPriorProgram:
    """Return independent priors for built-in asocial Q-value model."""

    return IndependentPriorProgram(
        {
            "alpha": uniform_log_prior(lower=0.0, upper=1.0),
            "beta": uniform_log_prior(lower=0.0, upper=20.0),
            "initial_value": uniform_log_prior(lower=-5.0, upper=5.0),
        }
    )


def test_sample_posterior_block_subject_study_data() -> None:
    """MCMC study-fitting helpers should produce finite aggregated outputs."""

    block = BlockData(
        block_id="b0",
        trials=(
            _trial(0, 1, 1.0),
            _trial(1, 0, 0.0),
            _trial(2, 1, 1.0),
        ),
    )
    subject = SubjectData(
        subject_id="s1",
        blocks=(block,),
    )
    study = StudyData(subjects=(subject,))

    block_result = sample_posterior_block_data(
        block,
        model_component_id="asocial_state_q_value_softmax",
        prior_program=_prior_program(),
        initial_params={"alpha": 0.4, "beta": 2.0, "initial_value": 0.0},
        n_samples=20,
        n_warmup=20,
        thin=1,
        proposal_scales={"alpha": 0.05, "beta": 0.2, "initial_value": 0.1},
        bounds={
            "alpha": (0.0, 1.0),
            "beta": (0.0, 20.0),
            "initial_value": (-5.0, 5.0),
        },
        random_seed=2,
    )
    assert block_result.block_id == "b0"
    assert block_result.n_trials == 3
    assert block_result.posterior_result.posterior_samples.n_draws == 20

    subject_result = sample_posterior_subject_data(
        subject,
        model_component_id="asocial_state_q_value_softmax",
        prior_program=_prior_program(),
        initial_params={"alpha": 0.4, "beta": 2.0, "initial_value": 0.0},
        n_samples=20,
        n_warmup=20,
        thin=1,
        proposal_scales={"alpha": 0.05, "beta": 0.2, "initial_value": 0.1},
        bounds={
            "alpha": (0.0, 1.0),
            "beta": (0.0, 20.0),
            "initial_value": (-5.0, 5.0),
        },
        random_seed=2,
    )
    assert subject_result.subject_id == "s1"
    assert len(subject_result.block_results) == 1
    assert math.isfinite(subject_result.total_map_log_likelihood)
    assert math.isfinite(subject_result.total_map_log_posterior)
    assert set(subject_result.mean_block_map_params) == {"alpha", "beta", "initial_value"}

    study_result = sample_posterior_study_data(
        study,
        model_component_id="asocial_state_q_value_softmax",
        prior_program=_prior_program(),
        initial_params={"alpha": 0.4, "beta": 2.0, "initial_value": 0.0},
        n_samples=20,
        n_warmup=20,
        thin=1,
        proposal_scales={"alpha": 0.05, "beta": 0.2, "initial_value": 0.1},
        bounds={
            "alpha": (0.0, 1.0),
            "beta": (0.0, 20.0),
            "initial_value": (-5.0, 5.0),
        },
        random_seed=2,
    )
    assert study_result.n_subjects == 1
    assert len(study_result.subject_results) == 1
    assert math.isfinite(study_result.total_map_log_likelihood)
    assert math.isfinite(study_result.total_map_log_posterior)
