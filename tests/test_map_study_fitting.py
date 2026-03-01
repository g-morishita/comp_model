"""Tests for study-level MAP fitting workflows."""

from __future__ import annotations

import math

from comp_model.core.data import BlockData, StudyData, SubjectData, TrialDecision
from comp_model.inference import (
    IndependentPriorProgram,
    MapFitSpec,
    fit_map_block_data,
    fit_map_study_data,
    fit_map_subject_data,
    normal_log_prior,
    uniform_log_prior,
)


def _trial(trial_index: int, action: int, reward: float) -> TrialDecision:
    """Build one trial row for MAP study-fitting tests."""

    return TrialDecision(
        trial_index=trial_index,
        decision_index=0,
        actor_id="subject",
        available_actions=(0, 1),
        action=action,
        observation={"state": 0},
        outcome={"reward": reward},
    )


def _fit_spec() -> MapFitSpec:
    """Return deterministic MAP fit specification for tests."""

    return MapFitSpec(
        estimator_type="scipy_map",
        initial_params={"alpha": 0.4, "beta": 1.0, "initial_value": 0.0},
        bounds={
            "alpha": (0.0, 1.0),
            "beta": (0.0, 20.0),
            "initial_value": (None, None),
        },
    )


def _prior_program() -> IndependentPriorProgram:
    """Return independent priors for built-in asocial Q-value model."""

    return IndependentPriorProgram(
        {
            "alpha": uniform_log_prior(lower=0.0, upper=1.0),
            "beta": uniform_log_prior(lower=0.0, upper=20.0),
            "initial_value": normal_log_prior(mean=0.0, std=1.0),
        }
    )


def test_fit_map_block_subject_study_data() -> None:
    """MAP study-fitting helpers should produce finite aggregated outputs."""

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

    block_result = fit_map_block_data(
        block,
        model_component_id="asocial_state_q_value_softmax",
        prior_program=_prior_program(),
        fit_spec=_fit_spec(),
    )
    assert block_result.block_id == "b0"
    assert block_result.n_trials == 3
    assert set(block_result.fit_result.map_params) == {"alpha", "beta", "initial_value"}

    subject_result = fit_map_subject_data(
        subject,
        model_component_id="asocial_state_q_value_softmax",
        prior_program=_prior_program(),
        fit_spec=_fit_spec(),
    )
    assert subject_result.subject_id == "s1"
    assert len(subject_result.block_results) == 1
    assert math.isfinite(subject_result.total_log_likelihood)
    assert math.isfinite(subject_result.total_log_posterior)
    assert set(subject_result.mean_map_params) == {"alpha", "beta", "initial_value"}

    study_result = fit_map_study_data(
        study,
        model_component_id="asocial_state_q_value_softmax",
        prior_program=_prior_program(),
        fit_spec=_fit_spec(),
    )
    assert study_result.n_subjects == 1
    assert len(study_result.subject_results) == 1
    assert math.isfinite(study_result.total_log_likelihood)
    assert math.isfinite(study_result.total_log_posterior)
