"""Tests for auto-dispatch config fitting helpers."""

from __future__ import annotations

import pytest

from comp_model.core.data import BlockData, StudyData, SubjectData, TrialDecision
from comp_model.inference import (
    fit_block_auto_from_config,
    fit_study_auto_from_config,
    fit_subject_auto_from_config,
    fit_trace_auto_from_config,
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
            "type": "mle",
            "solver": "grid_search",
            "parameter_grid": {
                "alpha": [0.3],
                "beta": [2.0],
                "initial_value": [0.0],
            },
        },
    }


def test_fit_auto_dispatches_mle_for_all_levels() -> None:
    """Auto-dispatch should route MLE configs for trace/block/subject/study inputs."""

    rows = (_trial(0, 1, 1.0), _trial(1, 0, 0.0), _trial(2, 1, 1.0))
    block = BlockData(block_id="b0", trials=rows)
    subject = SubjectData(subject_id="s1", blocks=(block,))
    study = StudyData(subjects=(subject,))

    assert hasattr(fit_trace_auto_from_config(rows, config=_mle_config()), "best")
    assert hasattr(fit_block_auto_from_config(block, config=_mle_config()).fit_result, "best")
    assert hasattr(fit_subject_auto_from_config(subject, config=_mle_config()), "block_results")
    assert hasattr(fit_study_auto_from_config(study, config=_mle_config()), "subject_results")


def test_fit_auto_rejects_unsupported_estimator_type() -> None:
    """Auto-dispatch should reject unsupported estimator types."""

    rows = (_trial(0, 1, 1.0),)
    config = {
        "model": {"component_id": "asocial_state_q_value_softmax", "kwargs": {}},
        "estimator": {"type": "scipy_map"},
    }

    with pytest.raises(ValueError, match="unsupported estimator.type"):
        fit_trace_auto_from_config(rows, config=config)
