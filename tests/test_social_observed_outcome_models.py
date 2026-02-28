"""Behavior tests for observed-outcome social model family."""

from __future__ import annotations

import pytest

from comp_model.core.contracts import DecisionContext
from comp_model.models import (
    SocialObservedOutcomeQModel,
    SocialObservedOutcomeQPerseverationModel,
    SocialObservedOutcomeValueShapingModel,
    SocialObservedOutcomeValueShapingPerseverationModel,
)


def _context(*, trial_index: int = 0, decision_index: int = 0) -> DecisionContext[int]:
    """Build a common 2-action context for tests."""

    return DecisionContext(
        trial_index=trial_index,
        available_actions=(0, 1),
        actor_id="subject",
        decision_index=decision_index,
    )


def test_observed_outcome_q_updates_only_demonstrator_stage() -> None:
    """Observed-outcome model should ignore subject outcomes and learn demo outcomes."""

    model = SocialObservedOutcomeQModel(alpha_observed=1.0, beta=1.0, initial_value=0.0)
    model.start_episode()

    context = _context()
    model.update(
        observation={"stage": "subject"},
        action=1,
        outcome={"reward": 1.0, "source_actor_id": "subject"},
        context=context,
    )
    assert model.q_values_snapshot(state=0) == {}

    model.update(
        observation={"stage": "demonstrator"},
        action=1,
        outcome={"reward": 1.0, "source_actor_id": "demonstrator"},
        context=context,
    )

    assert model.q_values_snapshot(state=0) == pytest.approx({0: 0.0, 1: 1.0})


def test_observed_outcome_q_perseveration_uses_last_subject_choice() -> None:
    """Perseveration variant should prefer repeated subject action when values equal."""

    model = SocialObservedOutcomeQPerseverationModel(alpha_observed=0.0, beta=0.0, kappa=3.0)
    model.start_episode()

    context = _context()
    model.update(
        observation={"stage": "subject"},
        action=1,
        outcome={"source_actor_id": "subject"},
        context=context,
    )

    distribution = model.action_distribution({"stage": "subject"}, context=context)
    assert distribution[1] > distribution[0]


def test_observed_outcome_value_shaping_applies_social_then_outcome_step() -> None:
    """Value-shaping model should apply both social and observed-outcome updates."""

    model = SocialObservedOutcomeValueShapingModel(
        alpha_observed=0.5,
        alpha_social=0.5,
        beta=1.0,
        pseudo_reward=1.0,
        initial_value=0.0,
    )
    model.start_episode()

    context = _context()
    model.update(
        observation={"stage": "demonstrator"},
        action=1,
        outcome={"reward": 1.0, "source_actor_id": "demonstrator"},
        context=context,
    )

    # 0 -> 0.5 (social), then 0.5 -> 0.75 (observed outcome)
    assert model.q_values_snapshot(state=0) == pytest.approx({0: 0.0, 1: 0.75})


def test_observed_outcome_value_shaping_perseveration_tracks_subject_choice() -> None:
    """Perseveration variant should bias toward last subject action."""

    model = SocialObservedOutcomeValueShapingPerseverationModel(
        alpha_observed=0.0,
        alpha_social=0.0,
        beta=0.0,
        kappa=2.0,
    )
    model.start_episode()

    context = _context()
    model.update(
        observation={"stage": "subject"},
        action=0,
        outcome={"source_actor_id": "subject"},
        context=context,
    )

    distribution = model.action_distribution({"stage": "subject"}, context=context)
    assert distribution[0] > distribution[1]
