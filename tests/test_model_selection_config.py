"""Tests for config-driven model-comparison helpers."""

from __future__ import annotations

import pytest

from comp_model.core.data import BlockData, StudyData, SubjectData, TrialDecision
from comp_model.inference import (
    compare_dataset_candidates_from_config,
    compare_study_candidates_from_config,
    compare_subject_candidates_from_config,
)


def _trial(trial_index: int, action: int, reward: float) -> TrialDecision:
    """Build one trial row for model-comparison config tests."""

    return TrialDecision(
        trial_index=trial_index,
        decision_index=0,
        actor_id="subject",
        available_actions=(0, 1),
        action=action,
        observation={"state": 0},
        outcome={"reward": reward},
    )


def test_compare_dataset_candidates_from_config_supports_mle() -> None:
    """Config model-comparison should select stronger MLE candidate."""

    rows = tuple(_trial(index, action=1, reward=1.0) for index in range(12))
    config = {
        "criterion": "log_likelihood",
        "candidates": [
            {
                "name": "good_mle",
                "model": {
                    "component_id": "asocial_state_q_value_softmax",
                    "kwargs": {},
                },
                "estimator": {
                    "type": "grid_search",
                    "parameter_grid": {
                        "alpha": [0.8],
                        "beta": [8.0],
                        "initial_value": [0.0],
                    },
                },
                "n_parameters": 3,
            },
            {
                "name": "bad_mle",
                "model": {
                    "component_id": "asocial_state_q_value_softmax",
                    "kwargs": {},
                },
                "estimator": {
                    "type": "grid_search",
                    "parameter_grid": {
                        "alpha": [0.2],
                        "beta": [0.0],
                        "initial_value": [0.0],
                    },
                },
                "n_parameters": 3,
            },
        ],
    }

    result = compare_dataset_candidates_from_config(rows, config=config)
    assert result.selected_candidate_name == "good_mle"
    assert len(result.comparisons) == 2


def test_compare_dataset_candidates_from_config_supports_map_candidates() -> None:
    """Config model-comparison should run MAP candidate definitions."""

    rows = tuple(_trial(index, action=1, reward=1.0) for index in range(8))
    config = {
        "criterion": "log_likelihood",
        "candidates": [
            {
                "name": "map_candidate",
                "model": {
                    "component_id": "asocial_state_q_value_softmax",
                    "kwargs": {},
                },
                "prior": {
                    "type": "independent",
                    "parameters": {
                        "alpha": {"distribution": "uniform", "lower": 0.0, "upper": 1.0},
                        "beta": {"distribution": "uniform", "lower": 0.0, "upper": 20.0},
                        "initial_value": {"distribution": "normal", "mean": 0.0, "std": 1.0},
                    },
                },
                "estimator": {
                    "type": "scipy_map",
                    "initial_params": {"alpha": 0.5, "beta": 1.0, "initial_value": 0.0},
                    "bounds": {
                        "alpha": [0.0, 1.0],
                        "beta": [0.0, 20.0],
                        "initial_value": [None, None],
                    },
                },
                "n_parameters": 3,
            },
            {
                "name": "mle_candidate",
                "model": {
                    "component_id": "asocial_state_q_value_softmax",
                    "kwargs": {},
                },
                "estimator": {
                    "type": "grid_search",
                    "parameter_grid": {
                        "alpha": [0.5],
                        "beta": [1.0],
                        "initial_value": [0.0],
                    },
                },
                "n_parameters": 3,
            },
        ],
    }

    result = compare_dataset_candidates_from_config(rows, config=config)
    assert len(result.comparisons) == 2
    assert {item.candidate_name for item in result.comparisons} == {"map_candidate", "mle_candidate"}


def test_compare_dataset_candidates_from_config_requires_prior_for_map() -> None:
    """MAP estimator candidates should fail without prior config."""

    rows = (_trial(0, action=1, reward=1.0),)
    config = {
        "candidates": [
            {
                "name": "map_candidate",
                "model": {
                    "component_id": "asocial_state_q_value_softmax",
                    "kwargs": {},
                },
                "estimator": {
                    "type": "scipy_map",
                    "initial_params": {"alpha": 0.5, "beta": 1.0, "initial_value": 0.0},
                    "bounds": {
                        "alpha": [0.0, 1.0],
                        "beta": [0.0, 20.0],
                        "initial_value": [None, None],
                    },
                },
            },
        ],
    }

    with pytest.raises(ValueError, match="prior is required"):
        compare_dataset_candidates_from_config(rows, config=config)


def test_compare_subject_and_study_candidates_from_config() -> None:
    """Config comparison helpers should support subject and study datasets."""

    block = BlockData(
        block_id="b0",
        trials=tuple(_trial(index, action=1, reward=1.0) for index in range(6)),
    )
    subject = SubjectData(subject_id="s1", blocks=(block,))
    study = StudyData(subjects=(subject,))

    config = {
        "criterion": "log_likelihood",
        "candidates": [
            {
                "name": "good_mle",
                "model": {
                    "component_id": "asocial_state_q_value_softmax",
                    "kwargs": {},
                },
                "estimator": {
                    "type": "grid_search",
                    "parameter_grid": {
                        "alpha": [0.8],
                        "beta": [8.0],
                        "initial_value": [0.0],
                    },
                },
                "n_parameters": 3,
            },
            {
                "name": "bad_mle",
                "model": {
                    "component_id": "asocial_state_q_value_softmax",
                    "kwargs": {},
                },
                "estimator": {
                    "type": "grid_search",
                    "parameter_grid": {
                        "alpha": [0.2],
                        "beta": [0.0],
                        "initial_value": [0.0],
                    },
                },
                "n_parameters": 3,
            },
        ],
    }

    subject_result = compare_subject_candidates_from_config(subject, config=config)
    assert subject_result.selected_candidate_name == "good_mle"

    study_result = compare_study_candidates_from_config(study, config=config)
    assert study_result.selected_candidate_name == "good_mle"
    assert study_result.n_subjects == 1
