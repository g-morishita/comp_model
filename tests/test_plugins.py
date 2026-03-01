"""Tests for plugin manifests and auto-discovery."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from comp_model.core.data import TrialDecision, trace_from_trial_decisions
from comp_model.demonstrators import (
    FixedSequenceDemonstrator,
    NoisyBestArmDemonstrator,
    RLDemonstrator,
)
from comp_model.generators import (
    EventTraceAsocialGenerator,
    EventTraceSocialPostOutcomeGenerator,
    EventTraceSocialPreChoiceGenerator,
    SocialBlockSpec,
)
from comp_model.inference import ActionReplayLikelihood, ActorSubsetReplayLikelihood
from comp_model.models import (
    AsocialQValueSoftmaxModel,
    AsocialStateQValueSoftmaxModel,
    AsocialStateQValueSoftmaxPerseverationModel,
    AsocialStateQValueSoftmaxSplitAlphaModel,
    SocialConstantDemoBiasObservedOutcomeQPerseverationModel,
    SocialDirichletReliabilityGatedDemoBiasObservedOutcomeQPerseverationModel,
    SocialObservedOutcomePolicyIndependentMixPerseverationModel,
    SocialObservedOutcomePolicySharedMixModel,
    SocialObservedOutcomePolicySharedMixPerseverationModel,
    SocialObservedOutcomeQModel,
    SocialObservedOutcomeQPerseverationModel,
    SocialObservedOutcomeValueShapingModel,
    SocialObservedOutcomeValueShapingPerseverationModel,
    SocialPolicyLearningOnlyModel,
    SocialPolicyLearningOnlyPerseverationModel,
    SocialPolicyReliabilityGatedDemoBiasObservedOutcomeQPerseverationModel,
    SocialPolicyReliabilityGatedValueShapingModel,
    SocialSelfOutcomeValueShapingModel,
    UniformRandomPolicyModel,
)
from comp_model.plugins import PluginRegistry, build_default_registry
from comp_model.problems import StationaryBanditProblem


def test_default_registry_discovers_builtin_components() -> None:
    """Default registry should include built-in components across all kinds."""

    registry = build_default_registry()

    model_ids = {manifest.component_id for manifest in registry.list(kind="model")}
    problem_ids = {manifest.component_id for manifest in registry.list(kind="problem")}
    demonstrator_ids = {manifest.component_id for manifest in registry.list(kind="demonstrator")}
    generator_ids = {manifest.component_id for manifest in registry.list(kind="generator")}

    assert {
        "asocial_q_value_softmax",
        "uniform_random_policy",
        "asocial_state_q_value_softmax",
        "asocial_state_q_value_softmax_perseveration",
        "asocial_state_q_value_softmax_split_alpha",
        "social_self_outcome_value_shaping",
        "social_observed_outcome_q",
        "social_observed_outcome_q_perseveration",
        "social_observed_outcome_value_shaping",
        "social_observed_outcome_value_shaping_perseveration",
        "social_policy_reliability_gated_value_shaping",
        "social_constant_demo_bias_observed_outcome_q_perseveration",
        "social_policy_reliability_gated_demo_bias_observed_outcome_q_perseveration",
        "social_dirichlet_reliability_gated_demo_bias_observed_outcome_q_perseveration",
        "social_observed_outcome_policy_shared_mix",
        "social_observed_outcome_policy_shared_mix_perseveration",
        "social_observed_outcome_policy_independent_mix_perseveration",
        "social_policy_learning_only",
        "social_policy_learning_only_perseveration",
    }.issubset(model_ids)
    assert {
        "stationary_bandit",
        "two_stage_social_bandit",
        "two_stage_social_post_outcome_bandit",
    }.issubset(problem_ids)
    assert {
        "fixed_sequence_demonstrator",
        "noisy_best_arm_demonstrator",
        "rl_demonstrator",
    }.issubset(demonstrator_ids)
    assert {
        "event_trace_asocial_generator",
        "event_trace_social_pre_choice_generator",
        "event_trace_social_post_outcome_generator",
    }.issubset(generator_ids)

    assert "q_learning" not in model_ids
    assert "random_agent" not in model_ids
    assert "qrl" not in model_ids
    assert "qrl_stay" not in model_ids
    assert "unidentifiable_qrl" not in model_ids


def test_registry_creates_components_from_factories() -> None:
    """Registry factories should construct usable instances."""

    registry = build_default_registry()

    instances = {
        "asocial_q_value_softmax": registry.create_model("asocial_q_value_softmax"),
        "uniform_random_policy": registry.create_model("uniform_random_policy"),
        "asocial_state_q_value_softmax": registry.create_model("asocial_state_q_value_softmax"),
        "asocial_state_q_value_softmax_perseveration": registry.create_model("asocial_state_q_value_softmax_perseveration"),
        "asocial_state_q_value_softmax_split_alpha": registry.create_model("asocial_state_q_value_softmax_split_alpha"),
        "social_self_outcome_value_shaping": registry.create_model("social_self_outcome_value_shaping"),
        "social_observed_outcome_q": registry.create_model("social_observed_outcome_q"),
        "social_observed_outcome_q_perseveration": registry.create_model("social_observed_outcome_q_perseveration"),
        "social_observed_outcome_value_shaping": registry.create_model("social_observed_outcome_value_shaping"),
        "social_observed_outcome_value_shaping_perseveration": registry.create_model("social_observed_outcome_value_shaping_perseveration"),
        "social_policy_reliability_gated_value_shaping": registry.create_model("social_policy_reliability_gated_value_shaping"),
        "social_constant_demo_bias_observed_outcome_q_perseveration": registry.create_model("social_constant_demo_bias_observed_outcome_q_perseveration"),
        "social_policy_reliability_gated_demo_bias_observed_outcome_q_perseveration": registry.create_model("social_policy_reliability_gated_demo_bias_observed_outcome_q_perseveration"),
        "social_dirichlet_reliability_gated_demo_bias_observed_outcome_q_perseveration": registry.create_model("social_dirichlet_reliability_gated_demo_bias_observed_outcome_q_perseveration"),
        "social_observed_outcome_policy_shared_mix": registry.create_model("social_observed_outcome_policy_shared_mix"),
        "social_observed_outcome_policy_shared_mix_perseveration": registry.create_model("social_observed_outcome_policy_shared_mix_perseveration"),
        "social_observed_outcome_policy_independent_mix_perseveration": registry.create_model("social_observed_outcome_policy_independent_mix_perseveration"),
        "social_policy_learning_only": registry.create_model("social_policy_learning_only"),
        "social_policy_learning_only_perseveration": registry.create_model("social_policy_learning_only_perseveration"),
    }

    assert isinstance(instances["asocial_q_value_softmax"], AsocialQValueSoftmaxModel)
    assert isinstance(instances["uniform_random_policy"], UniformRandomPolicyModel)
    assert isinstance(instances["asocial_state_q_value_softmax"], AsocialStateQValueSoftmaxModel)
    assert isinstance(
        instances["asocial_state_q_value_softmax_perseveration"],
        AsocialStateQValueSoftmaxPerseverationModel,
    )
    assert isinstance(instances["asocial_state_q_value_softmax_split_alpha"], AsocialStateQValueSoftmaxSplitAlphaModel)
    assert isinstance(instances["social_self_outcome_value_shaping"], SocialSelfOutcomeValueShapingModel)
    assert isinstance(instances["social_observed_outcome_q"], SocialObservedOutcomeQModel)
    assert isinstance(
        instances["social_observed_outcome_q_perseveration"],
        SocialObservedOutcomeQPerseverationModel,
    )
    assert isinstance(instances["social_observed_outcome_value_shaping"], SocialObservedOutcomeValueShapingModel)
    assert isinstance(
        instances["social_observed_outcome_value_shaping_perseveration"],
        SocialObservedOutcomeValueShapingPerseverationModel,
    )
    assert isinstance(
        instances["social_policy_reliability_gated_value_shaping"],
        SocialPolicyReliabilityGatedValueShapingModel,
    )
    assert isinstance(
        instances["social_constant_demo_bias_observed_outcome_q_perseveration"],
        SocialConstantDemoBiasObservedOutcomeQPerseverationModel,
    )
    assert isinstance(
        instances["social_policy_reliability_gated_demo_bias_observed_outcome_q_perseveration"],
        SocialPolicyReliabilityGatedDemoBiasObservedOutcomeQPerseverationModel,
    )
    assert isinstance(
        instances["social_dirichlet_reliability_gated_demo_bias_observed_outcome_q_perseveration"],
        SocialDirichletReliabilityGatedDemoBiasObservedOutcomeQPerseverationModel,
    )
    assert isinstance(instances["social_observed_outcome_policy_shared_mix"], SocialObservedOutcomePolicySharedMixModel)
    assert isinstance(
        instances["social_observed_outcome_policy_shared_mix_perseveration"],
        SocialObservedOutcomePolicySharedMixPerseverationModel,
    )
    assert isinstance(
        instances["social_observed_outcome_policy_independent_mix_perseveration"],
        SocialObservedOutcomePolicyIndependentMixPerseverationModel,
    )
    assert isinstance(instances["social_policy_learning_only"], SocialPolicyLearningOnlyModel)
    assert isinstance(instances["social_policy_learning_only_perseveration"], SocialPolicyLearningOnlyPerseverationModel)

    problem = registry.create_problem("stationary_bandit", reward_probabilities=[0.2, 0.8])
    fixed_demo = registry.create_demonstrator("fixed_sequence_demonstrator", sequence=[0, 1])
    noisy_demo = registry.create_demonstrator(
        "noisy_best_arm_demonstrator",
        reward_probabilities=[0.1, 0.9],
    )
    rl_demo = registry.create_demonstrator("rl_demonstrator", alpha=0.2, beta=2.0, initial_value=0.0)
    asocial_generator = registry.create_generator("event_trace_asocial_generator")
    pre_choice_generator = registry.create_generator("event_trace_social_pre_choice_generator")
    post_outcome_generator = registry.create_generator("event_trace_social_post_outcome_generator")

    assert isinstance(problem, StationaryBanditProblem)
    assert isinstance(fixed_demo, FixedSequenceDemonstrator)
    assert isinstance(noisy_demo, NoisyBestArmDemonstrator)
    assert isinstance(rl_demo, RLDemonstrator)
    assert isinstance(asocial_generator, EventTraceAsocialGenerator)
    assert isinstance(pre_choice_generator, EventTraceSocialPreChoiceGenerator)
    assert isinstance(post_outcome_generator, EventTraceSocialPostOutcomeGenerator)


def test_registry_rejects_removed_deprecated_model_ids() -> None:
    """Removed deprecated IDs should fail with KeyError."""

    registry = build_default_registry()

    with pytest.raises(KeyError):
        registry.create_model("q_learning")

    with pytest.raises(KeyError):
        registry.create_model("random_agent")

    with pytest.raises(KeyError):
        registry.create_model("qrl")


def test_discovery_is_idempotent_for_same_package() -> None:
    """Repeated discovery should not create duplicate IDs."""

    registry = PluginRegistry()
    registry.discover("comp_model.models")
    registry.discover("comp_model.models")

    manifests = registry.list(kind="model")
    ids = [manifest.component_id for manifest in manifests]
    assert len(ids) == len(set(ids))


def test_registry_smoke_instantiates_every_manifest() -> None:
    """Every discovered manifest should instantiate with smoke kwargs."""

    registry = build_default_registry()
    kwargs_by_key = _manifest_smoke_kwargs()

    for manifest in registry.list():
        kwargs = kwargs_by_key.get((manifest.kind, manifest.component_id), {})
        instance = registry.create(manifest.kind, manifest.component_id, **kwargs)
        assert instance is not None


def test_registry_model_likelihood_smoke_for_all_registered_models() -> None:
    """All registered models should evaluate likelihood on canonical traces."""

    registry = build_default_registry()
    asocial_trace, social_trace = _smoke_traces()

    asocial_likelihood = ActionReplayLikelihood()
    social_likelihood = ActorSubsetReplayLikelihood(
        fitted_actor_id="subject",
        scored_actor_ids=("subject",),
    )

    for manifest in registry.list("model"):
        model = registry.create_model(manifest.component_id)
        if manifest.component_id.startswith("social_"):
            replay = social_likelihood.evaluate(social_trace, model)
        else:
            replay = asocial_likelihood.evaluate(asocial_trace, model)
        assert np.isfinite(replay.total_log_likelihood)


def _manifest_smoke_kwargs() -> dict[tuple[str, str], dict[str, Any]]:
    """Return constructor kwargs for manifests requiring non-default args."""

    return {
        ("problem", "stationary_bandit"): {"reward_probabilities": [0.2, 0.8]},
        ("problem", "two_stage_social_bandit"): {"reward_probabilities": [0.2, 0.8]},
        ("problem", "two_stage_social_post_outcome_bandit"): {"reward_probabilities": [0.2, 0.8]},
        ("demonstrator", "fixed_sequence_demonstrator"): {"sequence": [0, 1]},
        ("demonstrator", "noisy_best_arm_demonstrator"): {"reward_probabilities": [0.2, 0.8]},
    }


def _smoke_traces():
    """Build deterministic asocial and social traces for model smoke checks."""

    asocial_trace = trace_from_trial_decisions(
        (
            TrialDecision(
                trial_index=0,
                decision_index=0,
                actor_id="subject",
                available_actions=(0, 1),
                action=1,
                observation={"state": 0},
                outcome={"reward": 1.0},
            ),
            TrialDecision(
                trial_index=1,
                decision_index=0,
                actor_id="subject",
                available_actions=(0, 1),
                action=0,
                observation={"state": 0},
                outcome={"reward": 0.0},
            ),
            TrialDecision(
                trial_index=2,
                decision_index=0,
                actor_id="subject",
                available_actions=(0, 1),
                action=1,
                observation={"state": 0},
                outcome={"reward": 1.0},
            ),
        )
    )

    pre_choice_generator = EventTraceSocialPreChoiceGenerator()
    social_block = pre_choice_generator.simulate_block(
        subject_model=UniformRandomPolicyModel(),
        demonstrator_model=UniformRandomPolicyModel(),
        block=SocialBlockSpec(
            n_trials=6,
            program_kwargs={"reward_probabilities": [0.2, 0.8]},
        ),
        rng=np.random.default_rng(123),
    )

    return asocial_trace, social_block.event_trace
