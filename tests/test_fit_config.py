"""Tests for config-driven fitting helpers."""

from __future__ import annotations

import math

import pytest

from comp_model.core.data import BlockData, StudyData, SubjectData, TrialDecision
from comp_model.demonstrators import FixedSequenceDemonstrator
from comp_model.inference import (
    fit_dataset_from_config,
    fit_spec_from_config,
    fit_study_from_config,
    fit_subject_from_config,
)
from comp_model.models import UniformRandomPolicyModel
from comp_model.problems import TwoStageSocialBanditProgram
from comp_model.runtime import SimulationConfig, run_trial_program


def _trial(trial_index: int, action: int, reward: float) -> TrialDecision:
    """Build one trial row for config-fitting tests."""

    return TrialDecision(
        trial_index=trial_index,
        decision_index=0,
        actor_id="subject",
        available_actions=(0, 1),
        action=action,
        observation={"state": 0},
        outcome={"reward": reward},
    )


def _social_trace(*, n_trials: int, seed: int):
    """Generate one two-actor social trace for config-fitting tests."""

    return run_trial_program(
        program=TwoStageSocialBanditProgram([0.5, 0.5]),
        models={
            "subject": UniformRandomPolicyModel(),
            "demonstrator": FixedSequenceDemonstrator(sequence=[1] * n_trials),
        },
        config=SimulationConfig(n_trials=n_trials, seed=seed),
    )


def test_fit_spec_from_config_grid_search() -> None:
    """Estimator config parser should construct FitSpec for grid-search."""

    spec = fit_spec_from_config(
        {
            "type": "grid_search",
            "parameter_grid": {
                "alpha": [0.2],
                "beta": [2.0],
                "initial_value": [0.0],
            },
        }
    )

    assert spec.estimator_type == "grid_search"
    assert spec.parameter_grid == {"alpha": [0.2], "beta": [2.0], "initial_value": [0.0]}


def test_fit_spec_from_config_rejects_unknown_transform() -> None:
    """Transform parser should fail on unsupported transform name."""

    with pytest.raises(ValueError, match="unsupported transform"):
        fit_spec_from_config(
            {
                "type": "transformed_scipy_minimize",
                "initial_params": {"alpha": 0.2},
                "transforms": {"alpha": "not_a_transform"},
            }
        )


def test_fit_spec_from_config_rejects_unknown_estimator_keys() -> None:
    """Estimator parser should reject unknown keys for known estimator types."""

    with pytest.raises(ValueError, match="estimator has unknown keys"):
        fit_spec_from_config(
            {
                "type": "grid_search",
                "parameter_grid": {"alpha": [0.2]},
                "unexpected": 1,
            }
        )


def test_fit_dataset_from_config_on_trial_rows() -> None:
    """Config-driven fit should run directly on TrialDecision rows."""

    rows = [_trial(0, 1, 1.0), _trial(1, 0, 0.0), _trial(2, 1, 1.0)]
    config = {
        "model": {
            "component_id": "asocial_state_q_value_softmax",
            "kwargs": {},
        },
        "estimator": {
            "type": "grid_search",
            "parameter_grid": {
                "alpha": [0.3],
                "beta": [2.0],
                "initial_value": [0.0],
            },
        },
    }

    result = fit_dataset_from_config(rows, config=config)
    assert result.best.params == {"alpha": 0.3, "beta": 2.0, "initial_value": 0.0}
    assert math.isfinite(result.best.log_likelihood)


def test_fit_dataset_from_config_rejects_unknown_top_level_keys() -> None:
    """Dataset config should fail fast on unknown top-level keys."""

    rows = [_trial(0, 1, 1.0), _trial(1, 0, 0.0)]
    config = {
        "model": {
            "component_id": "asocial_state_q_value_softmax",
            "kwargs": {},
        },
        "estimator": {
            "type": "grid_search",
            "parameter_grid": {
                "alpha": [0.3],
                "beta": [2.0],
                "initial_value": [0.0],
            },
        },
        "typo_field": True,
    }

    with pytest.raises(ValueError, match="config has unknown keys"):
        fit_dataset_from_config(rows, config=config)


def test_fit_study_from_config_runs_all_subjects() -> None:
    """Config-driven study fitting should evaluate every subject/block."""

    study = StudyData(
        subjects=(
            SubjectData(
                subject_id="s1",
                blocks=(
                    BlockData(block_id="b1", trials=(_trial(0, 1, 1.0), _trial(1, 1, 1.0))),
                ),
            ),
            SubjectData(
                subject_id="s2",
                blocks=(
                    BlockData(block_id="b2", trials=(_trial(0, 0, 0.0), _trial(1, 0, 0.0))),
                ),
            ),
        )
    )

    config = {
        "model": {
            "component_id": "asocial_state_q_value_softmax",
            "kwargs": {},
        },
        "estimator": {
            "type": "grid_search",
            "parameter_grid": {
                "alpha": [0.2],
                "beta": [1.5],
                "initial_value": [0.0],
            },
        },
    }

    result = fit_study_from_config(study, config=config)
    assert result.n_subjects == 2
    assert {subject.subject_id for subject in result.subject_results} == {"s1", "s2"}
    assert math.isfinite(result.total_log_likelihood)


def test_fit_subject_from_config_supports_joint_block_fit_strategy() -> None:
    """Subject config fitting should support joint block likelihood fitting."""

    subject = SubjectData(
        subject_id="s1",
        blocks=(
            BlockData(block_id="b1", trials=(_trial(0, 1, 1.0), _trial(1, 1, 1.0))),
            BlockData(block_id="b2", trials=(_trial(0, 0, 0.0), _trial(1, 0, 0.0))),
        ),
    )
    config = {
        "model": {
            "component_id": "asocial_state_q_value_softmax",
            "kwargs": {},
        },
        "estimator": {
            "type": "grid_search",
            "parameter_grid": {
                "alpha": [0.2],
                "beta": [1.5],
                "initial_value": [0.0],
            },
        },
        "block_fit_strategy": "joint",
    }

    result = fit_subject_from_config(subject, config=config)
    assert len(result.block_results) == 1
    assert result.block_results[0].block_id == "__joint__"
    assert result.block_results[0].n_trials == 4


def test_fit_subject_from_config_rejects_unknown_block_fit_strategy() -> None:
    """Subject config fitting should reject unknown block-fit strategy values."""

    subject = SubjectData(
        subject_id="s1",
        blocks=(BlockData(block_id="b1", trials=(_trial(0, 1, 1.0), _trial(1, 1, 1.0))),),
    )
    config = {
        "model": {
            "component_id": "asocial_state_q_value_softmax",
            "kwargs": {},
        },
        "estimator": {
            "type": "grid_search",
            "parameter_grid": {
                "alpha": [0.2],
                "beta": [1.5],
                "initial_value": [0.0],
            },
        },
        "block_fit_strategy": "not_a_strategy",
    }

    with pytest.raises(ValueError, match="config.block_fit_strategy must be one of"):
        fit_subject_from_config(subject, config=config)


def test_fit_dataset_from_config_supports_social_actor_subset_likelihood() -> None:
    """Config-driven fit should parse and use actor-subset likelihood on social traces."""

    trace = _social_trace(n_trials=12, seed=5)
    config = {
        "model": {
            "component_id": "asocial_state_q_value_softmax",
            "kwargs": {},
        },
        "estimator": {
            "type": "grid_search",
            "parameter_grid": {
                "alpha": [0.2],
                "beta": [1.0],
                "initial_value": [0.0],
            },
        },
        "likelihood": {
            "type": "actor_subset_replay",
            "fitted_actor_id": "subject",
            "scored_actor_ids": ["subject"],
            "auto_fill_unmodeled_actors": True,
        },
    }

    result = fit_dataset_from_config(trace, config=config)
    assert math.isfinite(result.best.log_likelihood)
