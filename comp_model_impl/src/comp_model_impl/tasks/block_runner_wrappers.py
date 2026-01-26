"""comp_model_impl.tasks.runner_block_wrapper

Runtime block-runner wrappers for bandit environments.

These wrappers implement the *trial-level interface schedule* (``TrialSpec``):

- per-trial outcome observation models (hidden / noisy / veridical)
- per-trial action availability (forced-choice)
- optional social observations via a demonstrator

They are intentionally lightweight and stateful.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from comp_model_core.errors import CompatibilityError
from comp_model_core.interfaces.bandit import BanditEnv
from comp_model_core.interfaces.block_runner import (
    BlockRunner,
    SocialBlockRunner,
    StepResult,
    SocialObservation,
)
from comp_model_core.interfaces.demonstrator import Demonstrator
from comp_model_core.spec import EnvironmentSpec, TrialSpec


def _socialize_spec(spec: EnvironmentSpec) -> EnvironmentSpec:
    """Return a copy of *spec* with ``is_social=True``."""

    return EnvironmentSpec(
        n_actions=int(spec.n_actions),
        outcome_type=spec.outcome_type,
        outcome_range=spec.outcome_range,
        outcome_is_bounded=bool(spec.outcome_is_bounded),
        is_social=True,
        state_kind=spec.state_kind,
        n_states=spec.n_states,
        state_shape=spec.state_shape,
    )


@dataclass(slots=True)
class BanditBlockRunner(BlockRunner):
    """Execute an asocial bandit block with a per-trial interface schedule."""

    env: BanditEnv
    trial_specs: Sequence[TrialSpec]

    def __post_init__(self) -> None:
        if bool(self.env.spec.is_social):
            raise CompatibilityError("BanditBlockRunner expects an asocial environment (env.spec.is_social=False)")
        if len(self.trial_specs) == 0:
            raise ValueError("trial_specs cannot be empty")
        self._state: Any = None

    @property
    def spec(self) -> EnvironmentSpec:
        return self.env.spec

    def reset(self, *, rng: np.random.Generator) -> Any:
        self._state = self.env.reset(rng=rng)
        return self._state

    def get_state(self) -> Any:
        return self.env.get_state()

    def trial_spec(self, *, t: int) -> TrialSpec:
        tt = int(t)
        if tt < 0 or tt >= len(self.trial_specs):
            raise IndexError(f"trial index out of range: t={tt}")
        return self.trial_specs[tt]

    def _check_action_allowed(self, *, t: int, action: int) -> None:
        ts = self.trial_spec(t=t)
        if ts.available_actions is None:
            return
        a = int(action)
        if a not in set(ts.available_actions):
            raise CompatibilityError(f"Trial {t}: action {a} not in available_actions={list(ts.available_actions)}")

    def step(self, *, t: int, action: int, rng: np.random.Generator) -> StepResult:
        self._check_action_allowed(t=t, action=action)

        env_step = self.env.step(action=int(action), rng=rng)
        outcome = float(env_step.outcome)
        ts = self.trial_spec(t=t)
        observed = ts.self_outcome.observe(true_outcome=outcome, env=self.env.spec, rng=rng)
        info = {} if env_step.info is None else dict(env_step.info)
        if ts.available_actions is not None:
            info.setdefault("available_actions", list(ts.available_actions))
        return StepResult(outcome=outcome, observed_outcome=observed, done=bool(env_step.done), info=info or None)


@dataclass(slots=True)
class SocialBanditBlockRunner(SocialBlockRunner):
    """Execute a social bandit block.

    The wrapped environment generates *true* outcomes. A demonstrator policy
    generates choices (and receives true outcomes). The subject/model receives a
    social observation with demonstrator choices and (optionally) demonstrator
    outcomes per the trial schedule.
    """

    env: BanditEnv
    demonstrator: Demonstrator
    trial_specs: Sequence[TrialSpec]

    def __post_init__(self) -> None:
        if bool(self.env.spec.is_social):
            raise CompatibilityError("SocialBanditBlockRunner expects an asocial base environment")
        if len(self.trial_specs) == 0:
            raise ValueError("trial_specs cannot be empty")
        self._state: Any = None
        self._subj_spec: EnvironmentSpec = _socialize_spec(self.env.spec)
        self._social_cache: list[SocialObservation | None] = [None] * len(self.trial_specs)

    @property
    def spec(self) -> EnvironmentSpec:
        return self._subj_spec

    def reset(self, *, rng: np.random.Generator) -> Any:
        # Environment state for the subject. In bandit tasks this is typically a
        # single context.
        self._state = self.env.reset(rng=rng)
        # Demonstrator should see the *environment* spec (asocial).
        self.demonstrator.reset(spec=self.env.spec, rng=rng)
        self._social_cache = [None] * len(self.trial_specs)
        return self._state

    def get_state(self) -> Any:
        return self.env.get_state()

    def trial_spec(self, *, t: int) -> TrialSpec:
        tt = int(t)
        if tt < 0 or tt >= len(self.trial_specs):
            raise IndexError(f"trial index out of range: t={tt}")
        return self.trial_specs[tt]

    def _check_action_allowed(self, *, t: int, action: int) -> None:
        ts = self.trial_spec(t=t)
        if ts.available_actions is None:
            return
        a = int(action)
        if a not in set(ts.available_actions):
            raise CompatibilityError(f"Trial {t}: action {a} not in available_actions={list(ts.available_actions)}")

    def observe_others(self, *, t: int, rng: np.random.Generator) -> SocialObservation:
        tt = int(t)
        cached = self._social_cache[tt]
        if cached is not None:
            return cached

        ts = self.trial_spec(t=t)
        state = self.env.get_state()

        # Demonstrator chooses based on the *environment* spec.
        demo_action = int(self.demonstrator.act(state=state, spec=self.env.spec, rng=rng))

        # Generate demonstrator outcome from the environment.
        demo_step = self.env.step(action=demo_action, rng=rng)
        demo_outcome_true = float(demo_step.outcome)

        # Demonstrator learns from the true outcome.
        self.demonstrator.update(
            state=state,
            action=demo_action,
            outcome=demo_outcome_true,
            spec=self.env.spec,
            rng=rng,
        )

        observed_demo = None
        if ts.demo_outcome is not None:
            observed_demo = ts.demo_outcome.observe(true_outcome=demo_outcome_true, env=self.env.spec, rng=rng)

        obs = SocialObservation(
            others_choices=[demo_action],
            others_outcomes=[demo_outcome_true],
            observed_others_outcomes=None if observed_demo is None else [float(observed_demo)],
            info=None,
        )
        self._social_cache[tt] = obs
        return obs

    def step(self, *, t: int, action: int, rng: np.random.Generator) -> StepResult:
        self._check_action_allowed(t=t, action=action)

        env_step = self.env.step(action=int(action), rng=rng)
        outcome = float(env_step.outcome)
        ts = self.trial_spec(t=t)
        observed = ts.self_outcome.observe(true_outcome=outcome, env=self.env.spec, rng=rng)
        info = {} if env_step.info is None else dict(env_step.info)
        if ts.available_actions is not None:
            info.setdefault("available_actions", list(ts.available_actions))
        return StepResult(outcome=outcome, observed_outcome=observed, done=bool(env_step.done), info=info or None)
