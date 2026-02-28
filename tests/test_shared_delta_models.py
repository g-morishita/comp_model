"""Tests for within-subject shared+delta wrappers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pytest

from comp_model.core.contracts import DecisionContext
from comp_model.models import (
    ConditionedSharedDeltaModel,
    ConditionedSharedDeltaSocialModel,
    SharedDeltaParameterSpec,
)


def _context() -> DecisionContext[int]:
    """Build default binary-action context used in tests."""

    return DecisionContext(trial_index=0, available_actions=(0, 1))


@dataclass
class FixedRightProbabilityModel:
    """Toy policy model with fixed probability of choosing action 1."""

    p_right: float

    def start_episode(self) -> None:
        """No-op episode reset."""

    def action_distribution(self, observation: Any, *, context: DecisionContext[int]) -> dict[int, float]:
        """Return fixed Bernoulli action distribution."""

        assert context.available_actions == (0, 1)
        return {0: 1.0 - float(self.p_right), 1: float(self.p_right)}

    def update(
        self,
        observation: Any,
        action: int,
        outcome: Any,
        *,
        context: DecisionContext[int],
    ) -> None:
        """No-op update."""



def _sigmoid(value: float) -> float:
    """Stable logistic transform from unconstrained z to (0, 1)."""

    return 1.0 / (1.0 + math.exp(-float(value)))


def test_conditioned_shared_delta_model_builds_condition_specific_policies() -> None:
    """Shared+delta wrapper should produce condition-specific constrained params."""

    wrapper = ConditionedSharedDeltaModel(
        model_factory=lambda params: FixedRightProbabilityModel(p_right=params["p_right"]),
        parameter_specs=(SharedDeltaParameterSpec(name="p_right", transform=_sigmoid),),
        conditions=("A", "B"),
        baseline_condition="A",
    )
    wrapper.set_params({"p_right__shared_z": 0.0, "p_right__delta_z__B": 2.0})
    wrapper.start_episode()

    wrapper.set_condition("A")
    dist_a = wrapper.action_distribution(observation={"condition": "A"}, context=_context())

    wrapper.set_condition("B")
    dist_b = wrapper.action_distribution(observation={"condition": "B"}, context=_context())

    assert dist_a[1] == pytest.approx(0.5)
    assert dist_b[1] == pytest.approx(_sigmoid(2.0))

    params_by_condition = wrapper.params_by_condition()
    assert params_by_condition["A"]["p_right"] == pytest.approx(0.5)
    assert params_by_condition["B"]["p_right"] == pytest.approx(_sigmoid(2.0))


def test_conditioned_shared_delta_model_auto_resolves_condition_from_observation() -> None:
    """Wrapper should switch active condition from observation payload."""

    wrapper = ConditionedSharedDeltaModel(
        model_factory=lambda params: FixedRightProbabilityModel(p_right=params["p_right"]),
        parameter_specs=(SharedDeltaParameterSpec(name="p_right", transform=_sigmoid),),
        conditions=("A", "B"),
        baseline_condition="A",
    )
    wrapper.set_params({"p_right__shared_z": 0.0, "p_right__delta_z__B": -2.0})
    wrapper.start_episode()

    dist = wrapper.action_distribution(observation={"condition": "B"}, context=_context())

    assert wrapper.active_condition == "B"
    assert dist[1] == pytest.approx(_sigmoid(-2.0))


def test_conditioned_shared_delta_model_rejects_incomplete_parameter_sets() -> None:
    """Wrapper should raise when shared+delta parameter keys are missing."""

    wrapper = ConditionedSharedDeltaModel(
        model_factory=lambda params: FixedRightProbabilityModel(p_right=params["p_right"]),
        parameter_specs=(SharedDeltaParameterSpec(name="p_right", transform=_sigmoid),),
        conditions=("A", "B"),
        baseline_condition="A",
    )

    with pytest.raises(ValueError, match=r"missing shared\+delta parameters"):
        wrapper.set_params({"p_right__shared_z": 0.0})



def test_conditioned_shared_delta_model_requires_set_params_before_start_episode() -> None:
    """Wrapper should require parameter initialization before episode start."""

    wrapper = ConditionedSharedDeltaModel(
        model_factory=lambda params: FixedRightProbabilityModel(p_right=params["p_right"]),
        parameter_specs=(SharedDeltaParameterSpec(name="p_right", transform=_sigmoid),),
        conditions=("A", "B"),
        baseline_condition="A",
    )

    with pytest.raises(ValueError, match=r"call set_params\(\) before start_episode\(\)"):
        wrapper.start_episode()


def test_conditioned_shared_delta_model_rejects_unknown_condition() -> None:
    """Explicit condition switching should reject unknown labels."""

    wrapper = ConditionedSharedDeltaModel(
        model_factory=lambda params: FixedRightProbabilityModel(p_right=params["p_right"]),
        parameter_specs=(SharedDeltaParameterSpec(name="p_right", transform=_sigmoid),),
        conditions=("A", "B"),
        baseline_condition="A",
    )
    wrapper.set_params({"p_right__shared_z": 0.0, "p_right__delta_z__B": 0.0})

    with pytest.raises(ValueError, match="unknown condition"):
        wrapper.set_condition("C")


def test_conditioned_shared_delta_social_model_delegates_like_asocial_wrapper() -> None:
    """Social wrapper variant should follow the same condition delegation semantics."""

    wrapper = ConditionedSharedDeltaSocialModel(
        model_factory=lambda params: FixedRightProbabilityModel(p_right=params["p_right"]),
        parameter_specs=(SharedDeltaParameterSpec(name="p_right", transform=_sigmoid),),
        conditions=("A", "B"),
        baseline_condition="A",
    )
    wrapper.set_params({"p_right__shared_z": 0.0, "p_right__delta_z__B": 2.0})
    wrapper.start_episode()

    dist = wrapper.action_distribution(observation={"condition": "B"}, context=_context())

    assert wrapper.active_condition == "B"
    assert dist[1] == pytest.approx(_sigmoid(2.0))
