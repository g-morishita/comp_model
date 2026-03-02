"""Tests for config-driven fit CLI."""

from __future__ import annotations

import json

import pytest

from comp_model.core.data import BlockData, StudyData, SubjectData, TrialDecision
from comp_model.inference import run_fit_cli
from comp_model.io import write_study_decisions_csv, write_trial_decisions_csv


def _trial(trial_index: int, action: int, reward: float) -> TrialDecision:
    """Build one trial row for fit CLI tests."""

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
            "type": "mle", "solver": "grid_search",
            "parameter_grid": {
                "alpha": [0.3, 0.5],
                "beta": [2.0],
                "initial_value": [0.0],
            },
        },
    }


def test_fit_cli_trial_mode_writes_summary(tmp_path, capsys) -> None:
    """Fit CLI should run on trial CSV input and write summary JSON."""

    trial_path = tmp_path / "trial.csv"
    write_trial_decisions_csv(
        (_trial(0, 1, 1.0), _trial(1, 0, 0.0), _trial(2, 1, 1.0)),
        trial_path,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_mle_config()), encoding="utf-8")

    code = run_fit_cli(
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
            "trial_fit",
        ]
    )
    captured = capsys.readouterr()

    assert code == 0
    assert "Fit complete" in captured.out
    summary_path = tmp_path / "trial_fit_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["input_kind"] == "trial"
    assert summary["level"] == "dataset"
    assert "best_params" in summary


def test_fit_cli_study_subject_mode_writes_summary(tmp_path, capsys) -> None:
    """Fit CLI should run on study CSV input at subject level."""

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
    study_path = tmp_path / "study.csv"
    write_study_decisions_csv(study, study_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_mle_config()), encoding="utf-8")

    code = run_fit_cli(
        [
            "--config",
            str(config_path),
            "--input-csv",
            str(study_path),
            "--input-kind",
            "study",
            "--level",
            "subject",
            "--subject-id",
            "s2",
            "--output-dir",
            str(tmp_path),
            "--prefix",
            "subject_fit",
        ]
    )
    captured = capsys.readouterr()

    assert code == 0
    assert "Fit complete" in captured.out
    summary_path = tmp_path / "subject_fit_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["input_kind"] == "study"
    assert summary["level"] == "subject"
    assert summary["subject_id"] == "s2"


def test_fit_cli_rejects_invalid_level_for_trial_input(tmp_path) -> None:
    """Fit CLI should reject unsupported level for trial input."""

    trial_path = tmp_path / "trial.csv"
    write_trial_decisions_csv((_trial(0, 1, 1.0),), trial_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_mle_config()), encoding="utf-8")

    with pytest.raises(ValueError, match="--level must be 'auto' or 'dataset'"):
        run_fit_cli(
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


def test_fit_cli_accepts_yaml_config(tmp_path, capsys) -> None:
    """Fit CLI should parse YAML config files."""

    trial_path = tmp_path / "trial.csv"
    write_trial_decisions_csv((_trial(0, 1, 1.0), _trial(1, 0, 0.0)), trial_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  component_id: asocial_state_q_value_softmax",
                "  kwargs: {}",
                "estimator:",
                "  type: mle",
                "  solver: grid_search",
                "  parameter_grid:",
                "    alpha: [0.3, 0.5]",
                "    beta: [2.0]",
                "    initial_value: [0.0]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    code = run_fit_cli(
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
            "trial_fit_yaml",
        ]
    )
    captured = capsys.readouterr()

    assert code == 0
    assert "Fit complete" in captured.out
    assert (tmp_path / "trial_fit_yaml_summary.json").exists()
