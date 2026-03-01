"""Tests for tabular CSV model-comparison adapters."""

from __future__ import annotations

import pytest

from comp_model.core.data import BlockData, StudyData, SubjectData, TrialDecision
from comp_model.inference import (
    compare_study_csv_candidates_from_config,
    compare_trial_csv_candidates_from_config,
)
from comp_model.io import write_study_decisions_csv, write_trial_decisions_csv


def _trial(trial_index: int, action: int, reward: float) -> TrialDecision:
    """Build one trial row for model-comparison tabular tests."""

    return TrialDecision(
        trial_index=trial_index,
        decision_index=0,
        actor_id="subject",
        available_actions=(0, 1),
        action=action,
        observation={"state": 0},
        outcome={"reward": reward},
    )


def _comparison_config() -> dict:
    """Build one deterministic MLE model-comparison config."""

    return {
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


def test_compare_trial_csv_candidates_from_config(tmp_path) -> None:
    """Trial CSV comparison adapter should select stronger candidate."""

    trial_path = tmp_path / "trial.csv"
    write_trial_decisions_csv(
        tuple(_trial(index, action=1, reward=1.0) for index in range(10)),
        trial_path,
    )

    result = compare_trial_csv_candidates_from_config(
        str(trial_path),
        config=_comparison_config(),
    )
    assert result.selected_candidate_name == "good_mle"
    assert len(result.comparisons) == 2


def test_compare_study_csv_candidates_from_config_study_level(tmp_path) -> None:
    """Study CSV adapter should support study-level comparison output."""

    study = StudyData(
        subjects=(
            SubjectData(
                subject_id="s1",
                blocks=(BlockData(block_id="b1", trials=tuple(_trial(i, 1, 1.0) for i in range(6))),),
            ),
            SubjectData(
                subject_id="s2",
                blocks=(BlockData(block_id="b1", trials=tuple(_trial(i, 1, 1.0) for i in range(6))),),
            ),
        )
    )
    study_path = tmp_path / "study.csv"
    write_study_decisions_csv(study, study_path)

    result = compare_study_csv_candidates_from_config(
        str(study_path),
        config=_comparison_config(),
        level="study",
    )
    assert result.selected_candidate_name == "good_mle"
    assert result.n_subjects == 2
    assert len(result.comparisons) == 2


def test_compare_study_csv_candidates_from_config_subject_selection(tmp_path) -> None:
    """Study CSV adapter should support explicit subject-level selection."""

    study = StudyData(
        subjects=(
            SubjectData(
                subject_id="s1",
                blocks=(BlockData(block_id="b1", trials=(_trial(0, 1, 1.0), _trial(1, 0, 0.0))),),
            ),
            SubjectData(
                subject_id="s2",
                blocks=(BlockData(block_id="b1", trials=(_trial(0, 1, 1.0), _trial(1, 1, 1.0))),),
            ),
        )
    )
    study_path = tmp_path / "study.csv"
    write_study_decisions_csv(study, study_path)

    result = compare_study_csv_candidates_from_config(
        str(study_path),
        config=_comparison_config(),
        level="subject",
        subject_id="s2",
    )
    assert result.subject_id == "s2"
    assert len(result.comparisons) == 2


def test_compare_study_csv_candidates_requires_subject_id_for_multi_subject(tmp_path) -> None:
    """Subject-level comparison should require subject ID for multi-subject CSV."""

    study = StudyData(
        subjects=(
            SubjectData(
                subject_id="s1",
                blocks=(BlockData(block_id="b1", trials=(_trial(0, 1, 1.0),)),),
            ),
            SubjectData(
                subject_id="s2",
                blocks=(BlockData(block_id="b1", trials=(_trial(0, 0, 0.0),)),),
            ),
        )
    )
    study_path = tmp_path / "study.csv"
    write_study_decisions_csv(study, study_path)

    with pytest.raises(ValueError, match="subject_id is required"):
        compare_study_csv_candidates_from_config(
            str(study_path),
            config=_comparison_config(),
            level="subject",
        )
