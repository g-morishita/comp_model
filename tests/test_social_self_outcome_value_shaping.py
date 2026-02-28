"""Tests for social self-outcome value-shaping model semantics."""

from __future__ import annotations

import pytest

from comp_model.core.contracts import DecisionContext
from comp_model.models import SocialSelfOutcomeValueShapingModel
from comp_model.problems import TwoStageSocialBanditProgram
from comp_model.runtime import SimulationConfig, run_trial_program


class _FixedActionModel:
    """Simple fixed-action actor for deterministic social test setup."""

    def __init__(self, action: int) -> None:
        self._action = action

    def start_episode(self) -> None:
        """No-op episode reset."""

    def action_distribution(self, observation: object, *, context: DecisionContext[int]) -> dict[int, float]:
        """Return deterministic one-hot distribution for configured action."""

        if self._action not in context.available_actions:
            raise ValueError("fixed action is not available")
        return {a: (1.0 if a == self._action else 0.0) for a in context.available_actions}

    def update(self, observation: object, action: int, outcome: object, *, context: DecisionContext[int]) -> None:
        """No-op update."""


def test_subject_distribution_reflects_same_trial_social_value_shaping() -> None:
    """Demonstrator action should shape values before subject decision probability."""

    model = SocialSelfOutcomeValueShapingModel(
        alpha_self=0.0,
        alpha_social=1.0,
        beta=8.0,
        kappa=0.0,
        pseudo_reward=1.0,
        initial_value=0.0,
    )
    program = TwoStageSocialBanditProgram(reward_probabilities=[0.5, 0.5])

    trace = run_trial_program(
        program=program,
        models={
            "demonstrator": _FixedActionModel(action=1),
            "subject": model,
        },
        config=SimulationConfig(n_trials=1, seed=0),
    )

    subject_decision = [
        event
        for event in trace.by_trial(0)
        if event.phase.value == "decision" and event.payload["actor_id"] == "subject"
    ][0]
    distribution = subject_decision.payload["distribution"]

    assert distribution[1] > 0.99
    assert distribution[1] > distribution[0]


def test_demonstrator_stage_does_not_trigger_private_outcome_update() -> None:
    """Private outcome learning must not run for demonstrator-stage observations."""

    model = SocialSelfOutcomeValueShapingModel(alpha_self=1.0, alpha_social=0.0, beta=1.0)
    model.start_episode()

    context = DecisionContext(trial_index=0, available_actions=(0, 1), actor_id="subject", decision_index=0)

    model.update(
        observation={"stage": "demonstrator"},
        action=1,
        outcome={"reward": 1.0},
        context=context,
    )

    assert model.q_values_snapshot(state=0) == {}


def test_social_shaping_is_idempotent_within_same_trial_and_node() -> None:
    """Repeated policy evaluation for same trial/node should not double-apply shaping."""

    model = SocialSelfOutcomeValueShapingModel(
        alpha_self=0.0,
        alpha_social=0.5,
        beta=1.0,
        kappa=0.0,
        pseudo_reward=1.0,
        initial_value=0.0,
    )
    model.start_episode()

    context = DecisionContext(
        trial_index=3,
        available_actions=(0, 1),
        actor_id="subject",
        decision_index=1,
    )
    observation = {
        "stage": "subject",
        "demonstrator_action": 1,
    }

    model.action_distribution(observation, context=context)
    q_after_first = model.q_values_snapshot(state=0)
    model.action_distribution(observation, context=context)
    q_after_second = model.q_values_snapshot(state=0)

    assert q_after_first == pytest.approx({0: 0.0, 1: 0.5})
    assert q_after_second == pytest.approx(q_after_first)
