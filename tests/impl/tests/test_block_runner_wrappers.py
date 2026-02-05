"""Tests for block runner wrappers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from comp_model_core.errors import CompatibilityError
from comp_model_core.interfaces.bandit import BanditEnv, EnvStep
from comp_model_core.interfaces.demonstrator import Demonstrator
from comp_model_core.spec import EnvironmentSpec, OutcomeType, OutcomeObservationKind, OutcomeObservationSpec, StateKind, TrialSpec

from comp_model_impl.bandits.bernoulli import BernoulliBanditEnv
from comp_model_impl.tasks.block_runner_wrappers import (
    BanditBlockRunner,
    SocialBanditBlockRunner,
    _socialize_spec,
)


@dataclass(slots=True)
class FixedOutcomeEnv(BanditEnv):
    """Bandit env returning a fixed outcome for testing."""

    outcome: float
    n_actions: int = 2
    _state: int = 0
    step_calls: int = 0

    @classmethod
    def from_config(cls, cfg):
        return cls(outcome=float(cfg["outcome"]))

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            n_actions=self.n_actions,
            outcome_type=OutcomeType.BINARY,
            outcome_range=(0.0, 1.0),
            outcome_is_bounded=True,
            is_social=False,
            state_kind=StateKind.DISCRETE,
            n_states=1,
        )

    def reset(self, *, rng: np.random.Generator) -> Any:
        self._state = 0
        return self._state

    def step(self, *, action: int, rng: np.random.Generator) -> EnvStep:
        self.step_calls += 1
        return EnvStep(outcome=float(self.outcome), done=False, info=None)

    def get_state(self) -> Any:
        return self._state


@dataclass(slots=True)
class SocialSpecEnv(FixedOutcomeEnv):
    """Bandit env with spec.is_social=True for error-path tests."""

    @property
    def spec(self) -> EnvironmentSpec:
        spec = super().spec
        return EnvironmentSpec(
            n_actions=spec.n_actions,
            outcome_type=spec.outcome_type,
            outcome_range=spec.outcome_range,
            outcome_is_bounded=spec.outcome_is_bounded,
            is_social=True,
            state_kind=spec.state_kind,
            n_states=spec.n_states,
            state_shape=spec.state_shape,
        )


@dataclass(slots=True)
class CountingDemonstrator(Demonstrator):
    """Demonstrator that always selects action 1 and records calls."""

    reset_calls: int = 0
    act_calls: int = 0
    update_calls: list[tuple[int, float]] = field(default_factory=list)

    def reset(self, *, spec: EnvironmentSpec, rng: np.random.Generator) -> None:
        self.reset_calls += 1

    def act(self, *, state: Any, spec: EnvironmentSpec, rng: np.random.Generator) -> int:
        self.act_calls += 1
        return 1

    def update(self, *, state: Any, action: int, outcome: float, spec: EnvironmentSpec, rng: np.random.Generator) -> None:
        self.update_calls.append((int(action), float(outcome)))


def test_socialize_spec_sets_social_flag() -> None:
    """_socialize_spec should set is_social=True and preserve other fields."""
    spec = EnvironmentSpec(
        n_actions=2,
        outcome_type=OutcomeType.BINARY,
        outcome_range=(0.0, 1.0),
        outcome_is_bounded=True,
        is_social=False,
        state_kind=StateKind.DISCRETE,
        n_states=1,
    )
    social = _socialize_spec(spec)
    assert social.is_social is True
    assert social.n_actions == spec.n_actions
    assert social.outcome_type == spec.outcome_type


def test_bandit_block_runner_enforces_action_availability_and_observation() -> None:
    """BanditBlockRunner should enforce available_actions and apply observation."""
    env = BernoulliBanditEnv(probs=[1.0, 0.0])
    trial_specs = [
        TrialSpec(
            self_outcome=OutcomeObservationSpec(kind=OutcomeObservationKind.VERIDICAL),
            available_actions=(0,),
        ),
    ]
    runner = BanditBlockRunner(env=env, trial_specs=trial_specs)
    rng = np.random.default_rng(0)
    runner.reset(rng=rng)

    with pytest.raises(CompatibilityError):
        runner.step(t=0, action=1, rng=rng)

    step = runner.step(t=0, action=0, rng=rng)
    assert step.outcome == 1.0
    assert step.observed_outcome == 1.0
    assert step.info is not None
    assert step.info["available_actions"] == [0]


def test_bandit_block_runner_rejects_social_env_and_empty_specs() -> None:
    """BanditBlockRunner should guard against invalid inputs."""
    with pytest.raises(CompatibilityError):
        BanditBlockRunner(env=SocialSpecEnv(outcome=1.0), trial_specs=[TrialSpec(self_outcome=OutcomeObservationSpec())])

    env = FixedOutcomeEnv(outcome=1.0)
    with pytest.raises(ValueError):
        BanditBlockRunner(env=env, trial_specs=[])


def test_social_bandit_block_runner_observe_others_caches_and_updates_demo() -> None:
    """SocialBanditBlockRunner should cache social observations per trial."""
    env = FixedOutcomeEnv(outcome=1.0)
    demo = CountingDemonstrator()
    trial_specs = [
        TrialSpec(
            self_outcome=OutcomeObservationSpec(kind=OutcomeObservationKind.VERIDICAL),
            demo_outcome=OutcomeObservationSpec(kind=OutcomeObservationKind.VERIDICAL),
        )
    ]
    runner = SocialBanditBlockRunner(env=env, demonstrator=demo, trial_specs=trial_specs)
    rng = np.random.default_rng(0)
    runner.reset(rng=rng)
    assert demo.reset_calls == 1

    obs1 = runner.observe_others(t=0, rng=rng)
    obs2 = runner.observe_others(t=0, rng=rng)
    assert obs1 is obs2
    assert demo.act_calls == 1
    assert env.step_calls == 1
    assert demo.update_calls == [(1, 1.0)]
    assert obs1.others_choices == [1]
    assert obs1.observed_others_outcomes == [1.0]

    step = runner.step(t=0, action=0, rng=rng)
    assert env.step_calls == 2
    assert step.outcome == 1.0


def test_social_bandit_block_runner_guards_inputs_and_hidden_demo() -> None:
    """SocialBanditBlockRunner should guard inputs and respect hidden demo outcomes."""
    with pytest.raises(CompatibilityError):
        SocialBanditBlockRunner(
            env=SocialSpecEnv(outcome=1.0),
            demonstrator=CountingDemonstrator(),
            trial_specs=[TrialSpec(self_outcome=OutcomeObservationSpec(), demo_outcome=OutcomeObservationSpec())],
        )

    env = FixedOutcomeEnv(outcome=0.0)
    with pytest.raises(ValueError):
        SocialBanditBlockRunner(env=env, demonstrator=CountingDemonstrator(), trial_specs=[])

    trial_specs = [
        TrialSpec(
            self_outcome=OutcomeObservationSpec(kind=OutcomeObservationKind.VERIDICAL),
            demo_outcome=OutcomeObservationSpec(kind=OutcomeObservationKind.HIDDEN),
        )
    ]
    runner = SocialBanditBlockRunner(env=env, demonstrator=CountingDemonstrator(), trial_specs=trial_specs)
    rng = np.random.default_rng(0)
    runner.reset(rng=rng)
    obs = runner.observe_others(t=0, rng=rng)
    assert obs.observed_others_outcomes is None
