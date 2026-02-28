"""Tests for multi-phase trial-program runtime semantics."""

from __future__ import annotations

import numpy as np
import pytest

from comp_model.core.events import EventPhase, validate_trace
from comp_model.models import RandomAgent
from comp_model.problems import TwoStageSocialBanditProgram
from comp_model.runtime import SimulationConfig, replay_trial_program, run_trial_program


def test_run_trial_program_emits_two_phase_blocks_per_trial() -> None:
    """Two-stage social program should emit two canonical phase blocks per trial."""

    program = TwoStageSocialBanditProgram(reward_probabilities=[0.2, 0.8])
    trace = run_trial_program(
        program=program,
        models={"demonstrator": RandomAgent(), "subject": RandomAgent()},
        config=SimulationConfig(n_trials=3, seed=9),
    )

    validate_trace(trace)
    assert len(trace.events) == 3 * 2 * 4

    for trial_index in range(3):
        trial_events = trace.by_trial(trial_index)
        phases = [event.phase for event in trial_events]
        assert phases == [
            EventPhase.OBSERVATION,
            EventPhase.DECISION,
            EventPhase.OUTCOME,
            EventPhase.UPDATE,
            EventPhase.OBSERVATION,
            EventPhase.DECISION,
            EventPhase.OUTCOME,
            EventPhase.UPDATE,
        ]

        decision_events = [event for event in trial_events if event.phase is EventPhase.DECISION]
        assert decision_events[0].payload["actor_id"] == "demonstrator"
        assert decision_events[1].payload["actor_id"] == "subject"


def test_subject_observation_includes_demonstrator_information() -> None:
    """Second node observation should include demonstrator action and outcome."""

    program = TwoStageSocialBanditProgram(reward_probabilities=[1.0, 0.0])
    trace = run_trial_program(
        program=program,
        models={"demonstrator": RandomAgent(), "subject": RandomAgent()},
        config=SimulationConfig(n_trials=1, seed=3),
    )

    trial_events = trace.by_trial(0)
    subject_observation_event = trial_events[4]
    observation = subject_observation_event.payload["observation"]

    assert observation["stage"] == "subject"
    assert observation["demonstrator_action"] in (0, 1)
    assert hasattr(observation["demonstrator_outcome"], "reward")


def test_replay_trial_program_supports_multi_actor_trace() -> None:
    """Replay should evaluate all decision nodes and track actor ownership."""

    program = TwoStageSocialBanditProgram(reward_probabilities=[0.5, 0.5])
    trace = run_trial_program(
        program=program,
        models={"demonstrator": RandomAgent(), "subject": RandomAgent()},
        config=SimulationConfig(n_trials=4, seed=1),
    )

    replay = replay_trial_program(
        trace=trace,
        models={"demonstrator": RandomAgent(), "subject": RandomAgent()},
    )

    assert len(replay.steps) == 8
    assert all(step.actor_id in {"demonstrator", "subject"} for step in replay.steps)
    assert all(step.learner_id == "subject" for step in replay.steps)

    expected = 8 * float(np.log(0.5))
    assert replay.total_log_likelihood == pytest.approx(expected)
