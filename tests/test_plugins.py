"""Tests for plugin manifests and auto-discovery."""

from __future__ import annotations

from comp_model_v2.models import QLearningAgent, RandomAgent
from comp_model_v2.plugins import PluginRegistry, build_default_registry
from comp_model_v2.problems import StationaryBanditProblem


def test_default_registry_discovers_builtin_components() -> None:
    """Default registry should include built-in models and problems."""

    registry = build_default_registry()

    model_ids = {manifest.component_id for manifest in registry.list(kind="model")}
    problem_ids = {manifest.component_id for manifest in registry.list(kind="problem")}

    assert {"q_learning", "random_agent"}.issubset(model_ids)
    assert {"stationary_bandit", "two_stage_social_bandit"}.issubset(problem_ids)


def test_registry_creates_components_from_factories() -> None:
    """Registry factories should construct usable instances."""

    registry = build_default_registry()

    model = registry.create_model("q_learning", alpha=0.1, beta=1.5, initial_value=0.25)
    random_model = registry.create_model("random_agent")
    problem = registry.create_problem("stationary_bandit", reward_probabilities=[0.2, 0.8])

    assert isinstance(model, QLearningAgent)
    assert isinstance(random_model, RandomAgent)
    assert isinstance(problem, StationaryBanditProblem)


def test_discovery_is_idempotent_for_same_package() -> None:
    """Repeated discovery should not create duplicate IDs."""

    registry = PluginRegistry()
    registry.discover("comp_model_v2.models")
    registry.discover("comp_model_v2.models")

    manifests = registry.list(kind="model")
    ids = [manifest.component_id for manifest in manifests]
    assert len(ids) == len(set(ids))
