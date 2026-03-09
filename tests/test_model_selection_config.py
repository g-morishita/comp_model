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


def _candidate(name: str, *, alpha: float, beta: float) -> dict:
    """Build one MLE candidate config."""

    return {
        "name": name,
        "model": {
            "component_id": "asocial_state_q_value_softmax",
            "kwargs": {},
        },
        "estimator": {
            "type": "mle",
            "solver": "grid_search",
            "parameter_grid": {
                "alpha": [alpha],
                "beta": [beta],
                "initial_value": [0.0],
            },
        },
        "n_parameters": 3,
    }


def test_compare_dataset_candidates_from_config_supports_mle() -> None:
    """Config model-comparison should select stronger MLE candidate."""

    rows = tuple(_trial(index, action=1, reward=1.0) for index in range(12))
    config = {
        "criterion": "log_likelihood",
        "candidates": [
            _candidate("good_mle", alpha=0.8, beta=8.0),
            _candidate("bad_mle", alpha=0.2, beta=0.0),
        ],
    }

    result = compare_dataset_candidates_from_config(rows, config=config)
    assert result.selected_candidate_name == "good_mle"
    assert len(result.comparisons) == 2


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
            _candidate("good_mle", alpha=0.8, beta=8.0),
            _candidate("bad_mle", alpha=0.2, beta=0.0),
        ],
    }

    subject_result = compare_subject_candidates_from_config(subject, config=config)
    assert subject_result.selected_candidate_name == "good_mle"

    study_result = compare_study_candidates_from_config(study, config=config)
    assert study_result.selected_candidate_name == "good_mle"
    assert study_result.n_subjects == 1


def test_compare_subject_candidates_from_config_supports_joint_block_strategy() -> None:
    """Subject/study config comparison should support joint block fitting mode."""

    subject = SubjectData(
        subject_id="s1",
        blocks=(
            BlockData(block_id="b0", trials=tuple(_trial(index, action=1, reward=1.0) for index in range(6))),
            BlockData(block_id="b1", trials=tuple(_trial(index, action=1, reward=1.0) for index in range(6))),
        ),
    )
    study = StudyData(subjects=(subject,))
    config = {
        "criterion": "log_likelihood",
        "block_fit_strategy": "joint",
        "candidates": [
            _candidate("good_mle", alpha=0.8, beta=8.0),
            _candidate("bad_mle", alpha=0.2, beta=0.0),
        ],
    }

    subject_result = compare_subject_candidates_from_config(subject, config=config)
    assert subject_result.selected_candidate_name == "good_mle"

    study_result = compare_study_candidates_from_config(study, config=config)
    assert study_result.selected_candidate_name == "good_mle"


def test_compare_dataset_candidates_from_config_rejects_unsupported_estimator() -> None:
    """Model-selection config should reject non-MLE estimator types."""

    rows = (_trial(0, action=1, reward=1.0),)
    config = {
        "candidates": [
            {
                "name": "bad",
                "model": {"component_id": "asocial_state_q_value_softmax", "kwargs": {}},
                "estimator": {"type": "scipy_map"},
            },
        ],
    }

    with pytest.raises(ValueError, match="estimator.type must be one of"):
        compare_dataset_candidates_from_config(rows, config=config)


def test_compare_dataset_candidates_from_config_rejects_unknown_top_level_keys() -> None:
    """Model-selection config should fail fast on unknown top-level keys."""

    rows = (_trial(0, action=1, reward=1.0),)
    config = {
        "extra": True,
        "candidates": [_candidate("mle", alpha=0.5, beta=1.0)],
    }

    with pytest.raises(ValueError, match="config has unknown keys"):
        compare_dataset_candidates_from_config(rows, config=config)


def test_compare_dataset_candidates_from_config_rejects_invalid_criterion() -> None:
    """Model-selection config should reject unsupported criteria."""

    rows = (_trial(0, action=1, reward=1.0),)
    config = {
        "criterion": "cross_validation",
        "candidates": [_candidate("mle", alpha=0.5, beta=1.0)],
    }

    with pytest.raises(ValueError, match="config.criterion must be one of"):
        compare_dataset_candidates_from_config(rows, config=config)
