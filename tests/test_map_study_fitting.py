"""Tests for removed study-level MAP fitting wrappers."""

from __future__ import annotations

import pytest

from comp_model.core.data import BlockData, SubjectData, TrialDecision
from comp_model.inference.bayes import IndependentPriorProgram, MapFitSpec, uniform_log_prior
from comp_model.inference.map_study_fitting import fit_map_block_data, fit_map_subject_data


def _trial(trial_index: int, action: int, reward: float) -> TrialDecision:
    """Build one trial row for map-study compatibility tests."""

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
    """Build one legacy MAP fit spec."""

    return MapFitSpec(
        estimator_type="scipy_map",
        initial_params={"alpha": 0.5},
        bounds={"alpha": (0.0, 1.0)},
    )


def _prior_program() -> IndependentPriorProgram:
    """Build one legacy independent prior."""

    return IndependentPriorProgram(
        {"alpha": uniform_log_prior(lower=0.0, upper=1.0)},
        require_all=False,
    )


def test_fit_map_block_data_raises_removed_runtime_error() -> None:
    """Legacy map-study block fitting should fail fast with guidance."""

    block = BlockData(
        block_id="b0",
        trials=(_trial(0, 1, 1.0), _trial(1, 0, 0.0)),
    )
    with pytest.raises(RuntimeError, match="no longer supported"):
        fit_map_block_data(
            block,
            model_component_id="asocial_state_q_value_softmax",
            prior_program=_prior_program(),
            fit_spec=_fit_spec(),
        )


def test_fit_map_subject_data_raises_removed_runtime_error() -> None:
    """Legacy map-study subject fitting should fail fast with guidance."""

    subject = SubjectData(
        subject_id="s1",
        blocks=(
            BlockData(block_id="b0", trials=(_trial(0, 1, 1.0),)),
            BlockData(block_id="b1", trials=(_trial(0, 0, 0.0),)),
        ),
    )
    with pytest.raises(RuntimeError, match="no longer supported"):
        fit_map_subject_data(
            subject,
            model_component_id="asocial_state_q_value_softmax",
            prior_program=_prior_program(),
            fit_spec=_fit_spec(),
            block_fit_strategy="joint",
        )

