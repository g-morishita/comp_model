"""Tests for parity benchmark CLI helper."""

from __future__ import annotations

import json
from pathlib import Path

from comp_model.analysis import run_parity_benchmark_cli
from comp_model.core.data import TrialDecision, trace_from_trial_decisions
from comp_model.inference import ActionReplayLikelihood
from comp_model.plugins import build_default_registry


def _trial(trial_index: int, action: int, reward: float) -> TrialDecision:
    """Build one trial decision row for parity CLI tests."""

    return TrialDecision(
        trial_index=trial_index,
        decision_index=0,
        actor_id="subject",
        available_actions=(0, 1),
        action=action,
        observation={"state": 0},
        outcome={"reward": reward},
    )


def _write_fixture(tmp_path: Path, expected_offset: float = 0.0) -> Path:
    """Write one-case parity fixture and return file path."""

    registry = build_default_registry()
    decisions = (_trial(0, 1, 1.0), _trial(1, 0, 0.0), _trial(2, 1, 1.0))
    trace = trace_from_trial_decisions(decisions)
    expected = ActionReplayLikelihood().evaluate(
        trace,
        registry.create_model(
            "asocial_state_q_value_softmax",
            alpha=0.3,
            beta=2.0,
            initial_value=0.0,
        ),
    ).total_log_likelihood
    fixture = {
        "cases": [
            {
                "name": "cli_case",
                "model_component_id": "asocial_state_q_value_softmax",
                "params": {"alpha": 0.3, "beta": 2.0, "initial_value": 0.0},
                "model_kwargs": {},
                "expected_log_likelihood": float(expected + expected_offset),
                "trial_decisions": [
                    {
                        "trial_index": int(item.trial_index),
                        "decision_index": int(item.decision_index),
                        "actor_id": item.actor_id,
                        "available_actions": list(item.available_actions),
                        "action": int(item.action),
                        "observation": item.observation,
                        "outcome": item.outcome,
                    }
                    for item in decisions
                ],
            }
        ]
    }
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")
    return path


def test_parity_cli_success_returns_zero_and_writes_csv(tmp_path, capsys) -> None:
    """CLI should return ``0`` when all cases pass tolerance checks."""

    fixture_path = _write_fixture(tmp_path, expected_offset=0.0)
    output_path = tmp_path / "report.csv"
    code = run_parity_benchmark_cli(
        [
            "--fixture",
            str(fixture_path),
            "--output-csv",
            str(output_path),
            "--atol",
            "1e-12",
            "--rtol",
            "1e-12",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "1/1 passed" in captured.out
    assert output_path.exists()
    assert "cli_case" in output_path.read_text(encoding="utf-8")


def test_parity_cli_failure_returns_one(tmp_path, capsys) -> None:
    """CLI should return ``1`` when any case fails tolerance checks."""

    fixture_path = _write_fixture(tmp_path, expected_offset=0.5)
    output_path = tmp_path / "report.csv"
    code = run_parity_benchmark_cli(
        [
            "--fixture",
            str(fixture_path),
            "--output-csv",
            str(output_path),
            "--atol",
            "1e-12",
            "--rtol",
            "1e-12",
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "failed=1" in captured.out
    assert output_path.exists()
