"""Tests for CSV-driven fitting entrypoints."""

from __future__ import annotations

import pytest

from comp_model.core.data import BlockData, StudyData, SubjectData, TrialDecision
from comp_model.inference import fit_study_csv_from_config, fit_trial_csv_from_config
from comp_model.io import write_study_decisions_csv, write_trial_decisions_csv


def _trial(trial_index: int, action: int, reward: float) -> TrialDecision:
    """Build one trial row for tabular-fitting tests."""

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
    """Build one minimal MLE config for asocial fitting."""

    return {
        "model": {"component_id": "asocial_state_q_value_softmax", "kwargs": {}},
        "estimator": {
            "type": "mle", "solver": "grid_search",
            "parameter_grid": {
                "alpha": [0.3, 0.5],
                "beta": [2.0],
                "initial_value": [0.0],
            },
        },
    }


def test_fit_trial_csv_from_config_runs_end_to_end(tmp_path) -> None:
    """Trial-level CSV helper should fit and return an MLE result."""

    rows = (_trial(0, 1, 1.0), _trial(1, 0, 0.0), _trial(2, 1, 1.0))
    csv_path = tmp_path / "trial_rows.csv"
    write_trial_decisions_csv(rows, csv_path)

    result = fit_trial_csv_from_config(str(csv_path), config=_mle_config())

    assert result.best.params["beta"] == pytest.approx(2.0)
    assert set(result.best.params) == {"alpha", "beta", "initial_value"}


def test_fit_study_csv_from_config_supports_study_level(tmp_path) -> None:
    """Study-level CSV helper should fit all subjects/blocks."""

    study = StudyData(
        subjects=(
            SubjectData(
                subject_id="s1",
                blocks=(
                    BlockData(
                        block_id="b1",
                        trials=(_trial(0, 1, 1.0), _trial(1, 0, 0.0), _trial(2, 1, 1.0)),
                    ),
                ),
            ),
        )
    )
    csv_path = tmp_path / "study_rows.csv"
    write_study_decisions_csv(study, csv_path)

    result = fit_study_csv_from_config(str(csv_path), config=_mle_config(), level="study")

    assert result.n_subjects == 1
    assert len(result.subject_results) == 1


def test_fit_study_csv_from_config_supports_subject_selection(tmp_path) -> None:
    """Subject-level CSV helper should select and fit the requested subject."""

    study = StudyData(
        subjects=(
            SubjectData(
                subject_id="s1",
                blocks=(BlockData(block_id="b1", trials=(_trial(0, 1, 1.0), _trial(1, 0, 0.0))),),
            ),
            SubjectData(
                subject_id="s2",
                blocks=(BlockData(block_id="b1", trials=(_trial(0, 0, 0.0), _trial(1, 1, 1.0))),),
            ),
        )
    )
    csv_path = tmp_path / "study_rows.csv"
    write_study_decisions_csv(study, csv_path)

    result = fit_study_csv_from_config(
        str(csv_path),
        config=_mle_config(),
        level="subject",
        subject_id="s2",
    )

    assert result.subject_id == "s2"
    assert len(result.block_results) == 1


def test_fit_study_csv_from_config_requires_subject_id_for_multi_subject_study(tmp_path) -> None:
    """Subject-level fitting should require ``subject_id`` when ambiguous."""

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
    csv_path = tmp_path / "study_rows.csv"
    write_study_decisions_csv(study, csv_path)

    with pytest.raises(ValueError, match="subject_id is required"):
        fit_study_csv_from_config(
            str(csv_path),
            config=_mle_config(),
            level="subject",
        )
