"""Tests for parity benchmark analysis helpers."""

from __future__ import annotations

import json

from comp_model.analysis import (
    ParityFixtureCase,
    load_parity_fixture_file,
    run_parity_benchmark,
    write_parity_benchmark_csv,
)
from comp_model.core.data import TrialDecision, trace_from_trial_decisions
from comp_model.inference import ActionReplayLikelihood
from comp_model.plugins import build_default_registry


def _trial(trial_index: int, action: int, reward: float) -> TrialDecision:
    """Build one trial row for parity benchmark tests."""

    return TrialDecision(
        trial_index=trial_index,
        decision_index=0,
        actor_id="subject",
        available_actions=(0, 1),
        action=action,
        observation={"state": 0},
        outcome={"reward": reward},
    )


def test_run_parity_benchmark_reports_pass_and_fail() -> None:
    """Parity runner should mark cases by tolerance pass/fail."""

    reg = build_default_registry()
    decisions = (_trial(0, 1, 1.0), _trial(1, 0, 0.0), _trial(2, 1, 1.0))
    trace = trace_from_trial_decisions(decisions)
    expected = ActionReplayLikelihood().evaluate(
        trace,
        reg.create_model(
            "asocial_state_q_value_softmax",
            alpha=0.3,
            beta=2.0,
            initial_value=0.0,
        ),
    ).total_log_likelihood

    result = run_parity_benchmark(
        [
            ParityFixtureCase(
                name="pass_case",
                model_component_id="asocial_state_q_value_softmax",
                params={"alpha": 0.3, "beta": 2.0, "initial_value": 0.0},
                expected_log_likelihood=float(expected),
                trial_decisions=decisions,
                model_kwargs={},
            ),
            ParityFixtureCase(
                name="fail_case",
                model_component_id="asocial_state_q_value_softmax",
                params={"alpha": 0.3, "beta": 2.0, "initial_value": 0.0},
                expected_log_likelihood=float(expected + 1.0),
                trial_decisions=decisions,
                model_kwargs={},
            ),
        ],
        atol=1e-12,
        rtol=1e-12,
    )

    assert result.n_cases == 2
    assert result.n_passed == 1
    assert result.n_failed == 1


def test_load_fixture_and_write_csv(tmp_path) -> None:
    """Fixture loader and CSV writer should round-trip benchmark rows."""

    fixture = {
        "cases": [
            {
                "name": "fixture_case",
                "model_component_id": "asocial_state_q_value_softmax",
                "params": {"alpha": 0.3, "beta": 2.0, "initial_value": 0.0},
                "model_kwargs": {},
                "expected_log_likelihood": -3.0,
                "trial_decisions": [
                    {
                        "trial_index": 0,
                        "decision_index": 0,
                        "actor_id": "subject",
                        "available_actions": [0, 1],
                        "action": 1,
                        "observation": {"state": 0},
                        "outcome": {"reward": 1.0},
                    }
                ],
            }
        ]
    }
    fixture_path = tmp_path / "parity_fixture.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")

    cases = load_parity_fixture_file(fixture_path)
    assert len(cases) == 1
    result = run_parity_benchmark(cases, atol=10.0, rtol=0.0)
    assert result.n_cases == 1

    out_path = write_parity_benchmark_csv(result, tmp_path / "parity_out.csv")
    assert out_path.exists()
    rows = out_path.read_text(encoding="utf-8")
    assert "fixture_case" in rows

