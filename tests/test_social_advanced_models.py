"""Behavior tests for advanced social model families."""

from __future__ import annotations

from comp_model.core.contracts import DecisionContext
from comp_model.models import (
    SocialConstantDemoBiasObservedOutcomeQPerseverationModel,
    SocialDirichletReliabilityGatedDemoBiasObservedOutcomeQPerseverationModel,
    SocialObservedOutcomePolicyIndependentMixPerseverationModel,
    SocialObservedOutcomePolicySharedMixModel,
    SocialObservedOutcomePolicySharedMixPerseverationModel,
    SocialPolicyLearningOnlyModel,
    SocialPolicyLearningOnlyPerseverationModel,
    SocialPolicyReliabilityGatedDemoBiasObservedOutcomeQPerseverationModel,
    SocialPolicyReliabilityGatedValueShapingModel,
)


def _context(*, trial_index: int = 0, decision_index: int = 0) -> DecisionContext[int]:
    """Build a common 2-action context for tests."""

    return DecisionContext(
        trial_index=trial_index,
        available_actions=(0, 1),
        actor_id="subject",
        decision_index=decision_index,
    )


def test_reliability_gated_value_shaping_depends_on_policy_reliability() -> None:
    """No policy learning means no reliability-gated shaping; full policy learning enables shaping."""

    context = _context()

    low_rel = SocialPolicyReliabilityGatedValueShapingModel(
        alpha_observed=0.0,
        alpha_social_base=1.0,
        alpha_policy=0.0,
        beta=1.0,
        kappa=0.0,
    )
    low_rel.start_episode()
    low_rel.update(
        observation={"stage": "demonstrator"},
        action=1,
        outcome={"source_actor_id": "demonstrator"},
        context=context,
    )

    high_rel = SocialPolicyReliabilityGatedValueShapingModel(
        alpha_observed=0.0,
        alpha_social_base=1.0,
        alpha_policy=1.0,
        beta=1.0,
        kappa=0.0,
    )
    high_rel.start_episode()
    high_rel.update(
        observation={"stage": "demonstrator"},
        action=1,
        outcome={"source_actor_id": "demonstrator"},
        context=context,
    )

    assert low_rel.q_values_snapshot(state=0) == {0: 0.0, 1: 0.0}
    assert high_rel.q_values_snapshot(state=0) == {0: 0.0, 1: 1.0}


def test_constant_demo_bias_prefers_recent_demo_action() -> None:
    """Constant demo-bias model should copy recent demo action when values are equal."""

    model = SocialConstantDemoBiasObservedOutcomeQPerseverationModel(
        alpha_observed=0.0,
        demo_bias=3.0,
        beta=0.0,
        kappa=0.0,
    )
    model.start_episode()

    context = _context()
    model.update(
        observation={"stage": "demonstrator"},
        action=1,
        outcome={"source_actor_id": "demonstrator"},
        context=context,
    )

    distribution = model.action_distribution({"stage": "subject"}, context=context)
    assert distribution[1] > distribution[0]


def test_policy_reliability_gated_demo_bias_prefers_demo_when_reliable() -> None:
    """Reliability-gated demo-bias model should copy after deterministic demo behavior."""

    model = SocialPolicyReliabilityGatedDemoBiasObservedOutcomeQPerseverationModel(
        alpha_observed=0.0,
        alpha_policy=1.0,
        demo_bias_rel=5.0,
        beta=0.0,
        kappa=0.0,
    )
    model.start_episode()

    context = _context()
    model.update(
        observation={"stage": "demonstrator"},
        action=0,
        outcome={"source_actor_id": "demonstrator"},
        context=context,
    )

    distribution = model.action_distribution({"stage": "subject"}, context=context)
    assert distribution[0] > distribution[1]


def test_dirichlet_reliability_demo_bias_strengthens_with_repetition() -> None:
    """Dirichlet reliability should increase copying bias after repeated same demo actions."""

    model = SocialDirichletReliabilityGatedDemoBiasObservedOutcomeQPerseverationModel(
        alpha_observed=0.0,
        demo_bias_rel=6.0,
        beta=0.0,
        kappa=0.0,
        demo_dirichlet_prior=1.0,
    )
    model.start_episode()

    context = _context()
    model.update(
        observation={"stage": "demonstrator"},
        action=1,
        outcome={"source_actor_id": "demonstrator"},
        context=context,
    )
    dist_after_one = model.action_distribution({"stage": "subject"}, context=context)

    model.update(
        observation={"stage": "demonstrator"},
        action=1,
        outcome={"source_actor_id": "demonstrator"},
        context=context,
    )
    dist_after_two = model.action_distribution({"stage": "subject"}, context=context)

    assert dist_after_two[1] > dist_after_one[1]


def test_shared_mix_policy_only_prefers_demo_action() -> None:
    """Shared-mix model with mix_weight=0 should follow learned policy signal."""

    model = SocialObservedOutcomePolicySharedMixModel(
        alpha_observed=0.0,
        alpha_policy=1.0,
        beta=5.0,
        mix_weight=0.0,
    )
    model.start_episode()

    context = _context()
    model.update(
        observation={"stage": "demonstrator"},
        action=1,
        outcome={"source_actor_id": "demonstrator"},
        context=context,
    )

    distribution = model.action_distribution({"stage": "subject"}, context=context)
    assert distribution[1] > distribution[0]


def test_shared_mix_perseveration_prefers_last_subject_choice_when_beta_zero() -> None:
    """Shared-mix perseveration model should rely on stay bias when beta is zero."""

    model = SocialObservedOutcomePolicySharedMixPerseverationModel(
        alpha_observed=0.0,
        alpha_policy=0.0,
        beta=0.0,
        mix_weight=0.5,
        kappa=3.0,
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


def test_independent_mix_uses_policy_weight() -> None:
    """Independent-mix model should follow policy drive when value weight is zero."""

    model = SocialObservedOutcomePolicyIndependentMixPerseverationModel(
        alpha_observed=0.0,
        alpha_policy=1.0,
        beta_q=0.0,
        beta_policy=5.0,
        kappa=0.0,
    )
    model.start_episode()

    context = _context()
    model.update(
        observation={"stage": "demonstrator"},
        action=1,
        outcome={"source_actor_id": "demonstrator"},
        context=context,
    )

    distribution = model.action_distribution({"stage": "subject"}, context=context)
    assert distribution[1] > distribution[0]


def test_policy_learning_only_follows_demo_actions() -> None:
    """Policy-learning-only model should copy demonstrator action tendencies."""

    model = SocialPolicyLearningOnlyModel(alpha_policy=1.0, beta=5.0)
    model.start_episode()

    context = _context()
    model.update(
        observation={"stage": "demonstrator"},
        action=0,
        outcome={"source_actor_id": "demonstrator"},
        context=context,
    )

    distribution = model.action_distribution({"stage": "subject"}, context=context)
    assert distribution[0] > distribution[1]


def test_policy_learning_only_perseveration_prefers_last_subject_choice_when_beta_zero() -> None:
    """Policy-only perseveration model should rely on stay bias when beta is zero."""

    model = SocialPolicyLearningOnlyPerseverationModel(alpha_policy=0.0, beta=0.0, kappa=3.0)
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
