"""Tests for auto-dispatch config fitting helpers."""

from __future__ import annotations

import math

import pytest

from comp_model.core.data import BlockData, StudyData, SubjectData, TrialDecision
from comp_model.inference import (
    fit_block_auto_from_config,
    fit_dataset_auto_from_config,
    fit_study_auto_from_config,
    fit_subject_auto_from_config,
)


def _trial(trial_index: int, action: int, reward: float) -> TrialDecision:
    """Build one trial row for auto-dispatch tests."""

    return TrialDecision(
        trial_index=trial_index,
        decision_index=0,
        actor_id="subject",
        available_actions=(0, 1),
        action=action,
        observation={"state": 0},
        outcome={"reward": reward},
    )


def _mle_config() -> dict:
    """Build one minimal MLE config."""

    return {
        "model": {"component_id": "asocial_state_q_value_softmax", "kwargs": {}},
        "estimator": {
            "type": "grid_search",
            "parameter_grid": {
                "alpha": [0.3],
                "beta": [2.0],
                "initial_value": [0.0],
            },
        },
    }


def _map_config() -> dict:
    """Build one minimal MAP config."""

    return {
        "model": {"component_id": "asocial_state_q_value_softmax", "kwargs": {}},
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
    }


def _hierarchical_config() -> dict:
    """Build one minimal within-subject hierarchical MAP config."""

    return {
        "model": {"component_id": "asocial_state_q_value_softmax", "kwargs": {}},
        "estimator": {
            "type": "within_subject_hierarchical_map",
            "parameter_names": ["alpha", "beta", "initial_value"],
            "transforms": {
                "alpha": "unit_interval_logit",
                "beta": "positive_log",
                "initial_value": "identity",
            },
            "initial_group_location": {"alpha": 0.4, "beta": 1.0, "initial_value": 0.0},
            "initial_group_scale": {"alpha": 0.5, "beta": 0.5, "initial_value": 0.5},
        },
    }


def test_fit_dataset_auto_dispatches_mle_and_map() -> None:
    """Dataset auto-dispatch should route to MLE and MAP fit paths."""

    rows = (_trial(0, 1, 1.0), _trial(1, 0, 0.0), _trial(2, 1, 1.0))

    mle_result = fit_dataset_auto_from_config(rows, config=_mle_config())
    assert hasattr(mle_result, "best")

    map_result = fit_dataset_auto_from_config(rows, config=_map_config())
    assert hasattr(map_result, "map_candidate")


def test_fit_block_subject_study_auto_dispatch() -> None:
    """Block/subject/study auto-dispatch should return finite fit outputs."""

    block = BlockData(
        block_id="b0",
        trials=(_trial(0, 1, 1.0), _trial(1, 0, 0.0), _trial(2, 1, 1.0)),
    )
    subject = SubjectData(subject_id="s1", blocks=(block,))
    study = StudyData(subjects=(subject,))

    block_map = fit_block_auto_from_config(block, config=_map_config())
    assert hasattr(block_map, "fit_result")

    subject_h = fit_subject_auto_from_config(subject, config=_hierarchical_config())
    assert hasattr(subject_h, "block_results")
    assert math.isfinite(subject_h.total_log_posterior)

    study_h = fit_study_auto_from_config(study, config=_hierarchical_config())
    assert study_h.n_subjects == 1
    assert math.isfinite(study_h.total_log_posterior)


def test_fit_auto_rejects_unsupported_estimator_type() -> None:
    """Auto-dispatch should reject unsupported estimator names."""

    rows = (_trial(0, 1, 1.0),)
    config = {
        "model": {"component_id": "asocial_state_q_value_softmax", "kwargs": {}},
        "estimator": {"type": "unsupported_estimator"},
    }

    with pytest.raises(ValueError, match="unsupported estimator.type"):
        fit_dataset_auto_from_config(rows, config=config)
