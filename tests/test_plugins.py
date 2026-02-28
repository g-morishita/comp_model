"""Tests for plugin manifests and auto-discovery."""

from __future__ import annotations

from comp_model.demonstrators import (
    FixedSequenceDemonstrator,
    NoisyBestArmDemonstrator,
    RLDemonstrator,
)
from comp_model.generators import (
    EventTraceAsocialGenerator,
    EventTraceSocialPostOutcomeGenerator,
    EventTraceSocialPreChoiceGenerator,
)
from comp_model.models import QLearningAgent, RandomAgent
from comp_model.plugins import PluginRegistry, build_default_registry
from comp_model.problems import StationaryBanditProblem


def test_default_registry_discovers_builtin_components() -> None:
    """Default registry should include built-in components across all kinds."""

    registry = build_default_registry()

    model_ids = {manifest.component_id for manifest in registry.list(kind="model")}
    problem_ids = {manifest.component_id for manifest in registry.list(kind="problem")}
    demonstrator_ids = {manifest.component_id for manifest in registry.list(kind="demonstrator")}
    generator_ids = {manifest.component_id for manifest in registry.list(kind="generator")}

    assert {"q_learning", "random_agent"}.issubset(model_ids)
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


def test_registry_creates_components_from_factories() -> None:
    """Registry factories should construct usable instances."""

    registry = build_default_registry()

    model = registry.create_model("q_learning", alpha=0.1, beta=1.5, initial_value=0.25)
    random_model = registry.create_model("random_agent")
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

    assert isinstance(model, QLearningAgent)
    assert isinstance(random_model, RandomAgent)
    assert isinstance(problem, StationaryBanditProblem)
    assert isinstance(fixed_demo, FixedSequenceDemonstrator)
    assert isinstance(noisy_demo, NoisyBestArmDemonstrator)
    assert isinstance(rl_demo, RLDemonstrator)
    assert isinstance(asocial_generator, EventTraceAsocialGenerator)
    assert isinstance(pre_choice_generator, EventTraceSocialPreChoiceGenerator)
    assert isinstance(post_outcome_generator, EventTraceSocialPostOutcomeGenerator)


def test_discovery_is_idempotent_for_same_package() -> None:
    """Repeated discovery should not create duplicate IDs."""

    registry = PluginRegistry()
    registry.discover("comp_model.models")
    registry.discover("comp_model.models")

    manifests = registry.list(kind="model")
    ids = [manifest.component_id for manifest in manifests]
    assert len(ids) == len(set(ids))
