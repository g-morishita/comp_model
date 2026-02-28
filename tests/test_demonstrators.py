"""Tests for built-in demonstrator components."""

from __future__ import annotations

import pytest

from comp_model.core.contracts import DecisionContext
from comp_model.demonstrators import (
    FixedSequenceDemonstrator,
    NoisyBestArmDemonstrator,
    RLDemonstrator,
)
from comp_model.models import UniformRandomPolicyModel
from comp_model.problems import TwoStageSocialBanditProgram
from comp_model.runtime import SimulationConfig, run_trial_program


def test_fixed_sequence_demonstrator_follows_trial_order() -> None:
    """Fixed sequence demonstrator should emit one-hot action by trial index."""

    demo = FixedSequenceDemonstrator(sequence=[1, 0, 1])

    for trial_index, expected_action in enumerate([1, 0, 1]):
        context = DecisionContext(trial_index=trial_index, available_actions=(0, 1), actor_id="demonstrator")
        distribution = demo.action_distribution(observation=None, context=context)
        assert distribution[expected_action] == pytest.approx(1.0)
        assert sum(distribution.values()) == pytest.approx(1.0)


def test_fixed_sequence_repeat_last_fallback() -> None:
    """Repeat-last fallback should extend sequence deterministically."""

    demo = FixedSequenceDemonstrator(sequence=[0, 1], fallback="repeat_last")
    context = DecisionContext(trial_index=5, available_actions=(0, 1), actor_id="demonstrator")
    distribution = demo.action_distribution(observation=None, context=context)

    assert distribution[1] == pytest.approx(1.0)


def test_noisy_best_arm_demonstrator_prefers_best_action() -> None:
    """Noisy best-arm policy should assign highest mass to best available arm."""

    demo = NoisyBestArmDemonstrator(reward_probabilities=[0.1, 0.9, 0.2], epsilon=0.2)
    context = DecisionContext(trial_index=0, available_actions=(0, 1, 2), actor_id="demonstrator")

    distribution = demo.action_distribution(observation=None, context=context)

    assert distribution[1] > distribution[0]
    assert distribution[1] > distribution[2]
    assert sum(distribution.values()) == pytest.approx(1.0)


def test_rl_demonstrator_updates_q_values_from_outcomes() -> None:
    """RL demonstrator should update Q-values after observing rewards."""

    demo = RLDemonstrator(alpha=0.5, beta=1.0, initial_value=0.0)
    demo.start_episode()

    context = DecisionContext(trial_index=0, available_actions=(0, 1), actor_id="demonstrator")
    demo.update(observation=None, action=1, outcome={"reward": 1.0}, context=context)

    q_values = demo.q_values_snapshot()
    assert q_values[1] == pytest.approx(0.5)


def test_demonstrators_work_as_runtime_actor_models() -> None:
    """Demonstrators should run as actor models inside multi-phase runtime."""

    program = TwoStageSocialBanditProgram(reward_probabilities=[0.2, 0.8])
    trace = run_trial_program(
        program=program,
        models={
            "demonstrator": FixedSequenceDemonstrator(sequence=[1, 1, 1]),
            "subject": UniformRandomPolicyModel(),
        },
        config=SimulationConfig(n_trials=3, seed=0),
    )

    decision_events = [event for event in trace.events if event.phase.value == "decision"]
    demo_actions = [event.payload["action"] for event in decision_events if event.payload["actor_id"] == "demonstrator"]
    assert demo_actions == [1, 1, 1]
