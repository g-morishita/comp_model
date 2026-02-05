"""Tests for within-subject shared+delta model wrappers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pytest

from comp_model_core.interfaces.model import ComputationalModel, SocialComputationalModel
from comp_model_core.params import Identity, ParamDef, ParameterSchema, Sigmoid
from comp_model_impl.models.within_subject_shared_delta import (
    ConditionedSharedDeltaModel,
    ConditionedSharedDeltaSocialModel,
    constrained_params_by_condition_from_z,
    flatten_params_by_condition,
    wrap_model_with_shared_delta_conditions,
)


@dataclass(slots=True)
class DummyModel(ComputationalModel):
    """Minimal asocial model for testing shared+delta wrappers."""

    _schema: ParameterSchema = field(init=False, repr=False)
    set_params_calls: list[dict[str, float]] = field(default_factory=list, init=False)
    reset_calls: int = field(default=0, init=False)
    last_update: tuple[Any, int, float | None] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._schema = ParameterSchema(
            params=(
                ParamDef(name="alpha", default=0.0, bound=None, transform=Sigmoid()),
                ParamDef(name="beta", default=0.0, bound=None, transform=Identity()),
            )
        )

    @property
    def param_schema(self) -> ParameterSchema:
        """Return the dummy model parameter schema."""
        return self._schema

    def supports(self, spec: Any) -> bool:
        """Accept any spec for testing."""
        return True

    def set_params(self, params: Mapping[str, Any], *, strict: bool = True, check_bounds: bool = False) -> None:
        """Record validated parameters after applying the base implementation."""
        super().set_params(params, strict=strict, check_bounds=check_bounds)
        self.set_params_calls.append(self.get_params())

    def reset_block(self, *, spec: Any) -> None:
        """Track reset calls."""
        self.reset_calls += 1

    def action_probs(self, *, state: Any, spec: Any) -> np.ndarray:
        """Return a fixed action distribution."""
        return np.array([0.5, 0.5], dtype=float)

    def update(self, *, state: Any, action: int, outcome: float | None, spec: Any, info: Any | None = None) -> None:
        """Record the last update tuple."""
        self.last_update = (state, int(action), outcome)


@dataclass(slots=True)
class DummySocialModel(SocialComputationalModel):
    """Minimal social model for testing shared+delta wrappers."""

    _schema: ParameterSchema = field(init=False, repr=False)
    social_updates: list[Any] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._schema = ParameterSchema(
            params=(ParamDef(name="alpha", default=0.0, bound=None, transform=Identity()),)
        )

    @property
    def param_schema(self) -> ParameterSchema:
        """Return the dummy social model parameter schema."""
        return self._schema

    def reset_block(self, *, spec: Any) -> None:
        """No-op reset for testing."""
        return

    def action_probs(self, *, state: Any, spec: Any) -> np.ndarray:
        """Return a fixed action distribution."""
        return np.array([0.5, 0.5], dtype=float)

    def update(self, *, state: Any, action: int, outcome: float | None, spec: Any, info: Any | None = None) -> None:
        """No-op update for testing."""
        return

    def social_update(self, *, state: Any, social: Any, spec: Any, info: Any | None = None) -> None:
        """Record social updates."""
        self.social_updates.append((state, social))


def test_conditioned_shared_delta_model_applies_expected_params() -> None:
    """Shared+delta z-parameters should map to constrained parameters per condition."""
    base = DummyModel()
    wrapper = ConditionedSharedDeltaModel(
        base_model=base,
        conditions=["A", "B", "A"],
        baseline_condition="A",
    )
    assert wrapper.conditions == ["A", "B"]

    wrapper.set_condition("B")
    params_z = {
        "alpha__shared_z": 0.0,
        "beta__shared_z": 0.5,
        "alpha__delta_z__B": 1.0,
        "beta__delta_z__B": -0.5,
    }
    wrapper.set_params(params_z)

    sig = Sigmoid()
    expected_a = {"alpha": sig.forward(0.0), "beta": 0.5}
    expected_b = {"alpha": sig.forward(1.0), "beta": 0.0}

    params_by_cond = wrapper.params_by_condition()
    assert np.isclose(params_by_cond["A"]["alpha"], expected_a["alpha"])
    assert np.isclose(params_by_cond["A"]["beta"], expected_a["beta"])
    assert np.isclose(params_by_cond["B"]["alpha"], expected_b["alpha"])
    assert np.isclose(params_by_cond["B"]["beta"], expected_b["beta"])

    assert base.set_params_calls
    assert np.isclose(base.set_params_calls[-1]["alpha"], expected_b["alpha"])
    assert np.isclose(base.set_params_calls[-1]["beta"], expected_b["beta"])


def test_conditioned_shared_delta_model_guards_and_switches() -> None:
    """Guards should trigger when conditions are missing or unknown."""
    base = DummyModel()
    wrapper = ConditionedSharedDeltaModel(
        base_model=base,
        conditions=["A", "B"],
        baseline_condition="A",
    )

    with pytest.raises(ValueError):
        wrapper.reset_block(spec=None)

    with pytest.raises(ValueError):
        wrapper.set_condition("C")

    params_z = {
        "alpha__shared_z": 0.0,
        "beta__shared_z": 0.0,
        "alpha__delta_z__B": 0.0,
        "beta__delta_z__B": 0.0,
    }
    wrapper.set_params(params_z)
    assert base.set_params_calls == []

    wrapper.set_condition("A")
    assert base.set_params_calls


def test_wrap_model_with_shared_delta_conditions_dispatches_by_sociality() -> None:
    """Wrapper selection should preserve whether the model is social."""
    asocial = DummyModel()
    social = DummySocialModel()

    wrapped_asocial = wrap_model_with_shared_delta_conditions(
        model=asocial,
        conditions=["A", "B"],
        baseline_condition="A",
    )
    assert isinstance(wrapped_asocial, ConditionedSharedDeltaModel)

    wrapped_social = wrap_model_with_shared_delta_conditions(
        model=social,
        conditions=["A", "B"],
        baseline_condition="A",
    )
    assert isinstance(wrapped_social, ConditionedSharedDeltaSocialModel)

    wrapped_social.set_condition("A")
    wrapped_social.set_params(
        {"alpha__shared_z": 0.0, "alpha__delta_z__B": 0.0}
    )
    wrapped_social.social_update(state=0, social={"choice": 1}, spec=None)
    assert social.social_updates


def test_constrained_params_by_condition_and_flatten_helpers() -> None:
    """Helper utilities should return stable, flattened outputs."""
    wrapper = ConditionedSharedDeltaModel(
        base_model=DummyModel(),
        conditions=["A", "B"],
        baseline_condition="A",
    )
    params_z = {
        "alpha__shared_z": 0.0,
        "beta__shared_z": 0.0,
        "alpha__delta_z__B": 0.0,
        "beta__delta_z__B": 0.0,
    }
    params_by_cond = constrained_params_by_condition_from_z(wrapper, params_z)
    assert set(params_by_cond.keys()) == {"A", "B"}

    flat = flatten_params_by_condition(params_by_cond)
    assert "alpha__A" in flat
    assert "alpha__B" in flat
