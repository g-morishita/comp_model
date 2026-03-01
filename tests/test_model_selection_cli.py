"""Tests for config-driven model-comparison CLI."""

from __future__ import annotations

import json

import pytest

from comp_model.core.data import BlockData, StudyData, SubjectData, TrialDecision
from comp_model.inference import run_model_comparison_cli
from comp_model.io import write_study_decisions_csv, write_trial_decisions_csv


def _trial(trial_index: int, action: int, reward: float) -> TrialDecision:
    """Build one trial row for model-comparison CLI tests."""

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


def test_model_comparison_cli_trial_mode_writes_outputs(tmp_path, capsys) -> None:
    """Model-comparison CLI should write trial-level CSV and summary artifacts."""

    trial_path = tmp_path / "trial.csv"
    write_trial_decisions_csv(
        tuple(_trial(index, action=1, reward=1.0) for index in range(10)),
        trial_path,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_comparison_config()), encoding="utf-8")

    code = run_model_comparison_cli(
        [
            "--config",
            str(config_path),
            "--input-csv",
            str(trial_path),
            "--input-kind",
            "trial",
            "--output-dir",
            str(tmp_path),
            "--prefix",
            "trial_cmp",
        ]
    )
    captured = capsys.readouterr()

    assert code == 0
    assert "Model comparison complete" in captured.out
    summary_path = tmp_path / "trial_cmp_summary.json"
    csv_path = tmp_path / "trial_cmp_dataset_comparison.csv"
    assert summary_path.exists()
    assert csv_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["input_kind"] == "trial"
    assert summary["level"] == "dataset"
    assert summary["selected_candidate_name"] == "good_mle"


def test_model_comparison_cli_study_mode_writes_aggregate_and_subject_csv(tmp_path, capsys) -> None:
    """Model-comparison CLI should write study aggregate and subject-level CSVs."""

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
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_comparison_config()), encoding="utf-8")

    code = run_model_comparison_cli(
        [
            "--config",
            str(config_path),
            "--input-csv",
            str(study_path),
            "--input-kind",
            "study",
            "--output-dir",
            str(tmp_path),
            "--prefix",
            "study_cmp",
        ]
    )
    captured = capsys.readouterr()

    assert code == 0
    assert "Model comparison complete" in captured.out
    summary_path = tmp_path / "study_cmp_summary.json"
    aggregate_csv = tmp_path / "study_cmp_study_comparison.csv"
    subject_csv = tmp_path / "study_cmp_study_subject_comparison.csv"
    assert summary_path.exists()
    assert aggregate_csv.exists()
    assert subject_csv.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["input_kind"] == "study"
    assert summary["level"] == "study"
    assert summary["n_subjects"] == 2
    assert summary["selected_candidate_name"] == "good_mle"


def test_model_comparison_cli_rejects_invalid_level_for_trial_input(tmp_path) -> None:
    """Model-comparison CLI should reject unsupported level for trial input."""

    trial_path = tmp_path / "trial.csv"
    write_trial_decisions_csv((_trial(0, 1, 1.0),), trial_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_comparison_config()), encoding="utf-8")

    with pytest.raises(ValueError, match="--level must be 'auto' or 'dataset'"):
        run_model_comparison_cli(
            [
                "--config",
                str(config_path),
                "--input-csv",
                str(trial_path),
                "--input-kind",
                "trial",
                "--level",
                "study",
            ]
        )
