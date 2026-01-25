"""comp_model_impl.tasks.social_wrapper

Concrete block-runner implementations for bandit environments.

This module provides executable runtime wrappers that apply **trial-level interface
rules** (available actions; outcome observation models such as hidden/noisy feedback)
on top of a bandit environment.

Two runtime runners are implemented:

- :class:`BanditBlockRunner`
  Asocial block execution with trial-varying interface constraints.

- :class:`SocialBanditBlockRunner`
  Adds a demonstrator channel to the same underlying environment, producing social
  observations in addition to self outcomes.

Notes
-----
The underlying environment is responsible only for generating **true outcomes**.
Observation rules are implemented here via
:class:`comp_model_core.spec.OutcomeObservationSpec`.

See Also
--------
comp_model_core.interfaces.block_runner.BlockRunner
comp_model_core.interfaces.block_runner.SocialBlockRunner
comp_model_core.spec.TrialSpec
comp_model_core.spec.ResolvedTrialSpec
comp_model_core.spec.OutcomeObservationSpec
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from copy import deepcopy

import numpy as np

from comp_model_core.interfaces.bandit import BanditEnv
from comp_model_core.interfaces.demonstrator import Demonstrator
from comp_model_core.interfaces.block_runner import BlockRunner, SocialBlockRunner, StepResult, SocialObservation
from comp_model_core.spec import (
    EnvironmentSpec,
    OutcomeObservationKind,
    OutcomeObservationSpec,
    TrialSpec,
    ResolvedTrialSpec,
)


def parse_trial_specs(
    *,
    n_trials: int,
    raw_trial_specs: Sequence[Mapping[str, Any]] | None,
    default_self_outcome: OutcomeObservationSpec,
    default_demo_outcome: OutcomeObservationSpec | None,
) -> list[ResolvedTrialSpec]:
    """Parse and resolve plan-provided trial specs into resolved specs.

    Parameters
    ----------
    n_trials : int
        Number of trials in the block plan.
    raw_trial_specs : sequence of mapping or None
        Per-trial dictionaries (YAML/JSON style). If None, all trials use defaults.
        If provided, must have length ``n_trials``.
    default_self_outcome : OutcomeObservationSpec
        Block-level default observation model for the subject's own outcomes.
    default_demo_outcome : OutcomeObservationSpec or None
        Block-level default observation model for demonstrator outcomes as seen by
        the subject. Use None for asocial blocks.

    Returns
    -------
    list of ResolvedTrialSpec
        Per-trial resolved interface specifications with defaults applied.

    Raises
    ------
    ValueError
        If ``raw_trial_specs`` is provided but its length does not equal ``n_trials``.
    """
    if raw_trial_specs is None:
        trial_specs = [TrialSpec() for _ in range(int(n_trials))]
    else:
        if len(raw_trial_specs) != int(n_trials):
            raise ValueError("trial_specs length must equal n_trials")

        trial_specs = []
        for d in raw_trial_specs:
            trial_specs.append(
                TrialSpec(
                    self_outcome=d.get("self_outcome", None),
                    available_actions=d.get("available_actions", None),
                    demo_outcome=d.get("demo_outcome", None),
                    metadata=d.get("metadata", None),
                )
            )

    return [
        ts.resolve(default_self_outcome=default_self_outcome, default_demo_outcome=default_demo_outcome)
        for ts in trial_specs
    ]


def _obs_spec_to_dict(spec: OutcomeObservationSpec | None) -> dict[str, Any] | None:
    """Convert an outcome-observation spec into a JSON-friendly dict.

    This is used to embed the applied observation model into ``StepResult.info`` or
    social observation metadata.

    Parameters
    ----------
    spec : OutcomeObservationSpec or None
        Observation model specification.

    Returns
    -------
    dict[str, Any] or None
        Dict containing keys like ``kind``, and (when applicable) ``sigma`` or
        ``flip_p``. Returns None if ``spec`` is None.
    """
    if spec is None:
        return None
    out: dict[str, Any] = {"kind": spec.kind.name}
    if spec.kind is OutcomeObservationKind.GAUSSIAN:
        out["sigma"] = spec.sigma
    if spec.kind is OutcomeObservationKind.FLIP:
        out["flip_p"] = spec.flip_p
    out["clip_to_range"] = spec.clip_to_range
    return out


@dataclass(slots=True)
class BanditBlockRunner(BlockRunner):
    """Executable runner for an asocial bandit block.

    Parameters
    ----------
    env : BanditEnv
        Underlying environment used to generate true outcomes.
    resolved_trial_specs : Sequence[ResolvedTrialSpec]
        Trial-by-trial resolved interface specs (available actions; self outcome
        observation model; metadata). Length must match the plan's ``n_trials``.

    Notes
    -----
    - This is a **stateful runtime** object (created from a plan).
    - Outcome "visibility" is generalized into an outcome observation model:
      hidden, veridical, or noisy.
    - The environment is deep-copied in ``__post_init__`` to avoid state leakage if
      builders reuse environment instances.
    """

    env: BanditEnv
    resolved_trial_specs: Sequence[ResolvedTrialSpec]

    def __post_init__(self) -> None:
        # Deepcopy env to avoid state leakage across blocks when builders reuse objects.
        self.env = deepcopy(self.env)
        self._resolved: list[ResolvedTrialSpec] = list(self.resolved_trial_specs)

    @property
    def spec(self) -> EnvironmentSpec:
        """Environment contract for this runner."""
        return self.env.spec

    def reset(self, *, rng: np.random.Generator) -> Any:
        """Reset the underlying environment."""
        return self.env.reset(rng=rng)

    def get_state(self) -> Any:
        """Return the current environment state/context."""
        return self.env.get_state()

    def resolved_trial_spec(self, *, t: int) -> ResolvedTrialSpec:
        """Return resolved interface spec for trial ``t``."""
        return self._resolved[int(t)]

    def step(self, *, t: int, action: int, rng: np.random.Generator) -> StepResult:
        """Execute one trial: enforce action set, step env, apply observation model.

        Parameters
        ----------
        t : int
            Trial index (0-based).
        action : int
            Action selected by the agent/model.
        rng : numpy.random.Generator
            RNG used for stochastic environment outcomes and observation noise.

        Returns
        -------
        StepResult
            True outcome and observed outcome (possibly hidden/noisy), plus metadata.

        Raises
        ------
        ValueError
            If ``action`` is not allowed by the trial's available action set.
        """
        rts = self.resolved_trial_spec(t=t)

        # Enforce trial-level action availability.
        if rts.available_actions is not None and int(action) not in set(rts.available_actions):
            raise ValueError(f"Action {int(action)} not allowed on trial {int(t)}")

        estep = self.env.step(action=int(action), rng=rng)
        true_outcome = float(estep.outcome)

        # Apply the observation model (hidden / noisy / etc.).
        observed = rts.self_outcome.observe(true_outcome=true_outcome, env=self.spec, rng=rng)

        info = dict(estep.info or {})
        info.update(
            {
                "available_actions": None if rts.available_actions is None else list(rts.available_actions),
                "self_outcome_obs": _obs_spec_to_dict(rts.self_outcome),
            }
        )
        return StepResult(
            outcome=true_outcome,
            observed_outcome=None if observed is None else float(observed),
            done=bool(estep.done),
            info=info,
        )


@dataclass(slots=True)
class SocialBanditBlockRunner(SocialBlockRunner):
    """Bandit block runner with a demonstrator (social) channel.

    Parameters
    ----------
    env : BanditEnv
        Underlying environment used to generate outcomes.
    demonstrator : Demonstrator
        Demonstrator policy used to generate other-agent choices and outcomes.
    resolved_trial_specs : Sequence[ResolvedTrialSpec]
        Trial-by-trial resolved interface specs. For social blocks, this should
        include ``demo_outcome`` observation models (or defaults).

    Notes
    -----
    The underlying environment does not need to be intrinsically "social".
    This runner executes a demonstrator policy *inside the same environment* to
    produce social observations.

    Visibility/noise rules
    ----------------------
    - The demonstrator is assumed to observe its own **true** outcome for its own update.
    - The subject's observation of demonstrator outcomes is controlled by the per-trial
      ``ResolvedTrialSpec.demo_outcome`` observation model (hidden/noisy/veridical).
    """

    env: BanditEnv
    demonstrator: Demonstrator
    resolved_trial_specs: Sequence[ResolvedTrialSpec]

    def __post_init__(self) -> None:
        self.env = deepcopy(self.env)
        self.demonstrator = deepcopy(self.demonstrator)
        self._resolved: list[ResolvedTrialSpec] = list(self.resolved_trial_specs)

        # Build a "social" view of the environment spec for compatibility checks.
        s = self.env.spec
        self._spec = EnvironmentSpec(
            n_actions=s.n_actions,
            outcome_type=s.outcome_type,
            outcome_range=s.outcome_range,
            outcome_is_bounded=s.outcome_is_bounded,
            is_social=True,
            state_kind=s.state_kind,
            n_states=s.n_states,
            state_shape=s.state_shape,
        )

    @property
    def spec(self) -> EnvironmentSpec:
        """Environment contract for this runner (forced to is_social=True)."""
        return self._spec

    def reset(self, *, rng: np.random.Generator) -> Any:
        """Reset environment and demonstrator state."""
        st = self.env.reset(rng=rng)
        self.demonstrator.reset(spec=self.spec, rng=rng)
        return st

    def get_state(self) -> Any:
        """Return the current environment state/context."""
        return self.env.get_state()

    def resolved_trial_spec(self, *, t: int) -> ResolvedTrialSpec:
        """Return resolved interface spec for trial ``t``."""
        return self._resolved[int(t)]

    def step(self, *, t: int, action: int, rng: np.random.Generator) -> StepResult:
        """Execute one trial: enforce action set, step env, apply self observation model.

        Parameters
        ----------
        t : int
            Trial index (0-based).
        action : int
            Action selected by the agent/model.
        rng : numpy.random.Generator
            RNG used for stochastic environment outcomes and observation noise.

        Returns
        -------
        StepResult
            True outcome and observed outcome (possibly hidden/noisy), plus metadata.

        Raises
        ------
        ValueError
            If ``action`` is not allowed by the trial's available action set.
        """
        rts = self.resolved_trial_spec(t=t)
        if rts.available_actions is not None and int(action) not in set(rts.available_actions):
            raise ValueError(f"Action {int(action)} not allowed on trial {int(t)}")

        estep = self.env.step(action=int(action), rng=rng)
        true_outcome = float(estep.outcome)
        observed = rts.self_outcome.observe(true_outcome=true_outcome, env=self.spec, rng=rng)

        info = dict(estep.info or {})
        info.update(
            {
                "available_actions": None if rts.available_actions is None else list(rts.available_actions),
                "self_outcome_obs": _obs_spec_to_dict(rts.self_outcome),
            }
        )
        return StepResult(
            outcome=true_outcome,
            observed_outcome=None if observed is None else float(observed),
            done=bool(estep.done),
            info=info,
        )

    def observe_others(self, *, t: int, rng: np.random.Generator) -> SocialObservation:
        """Generate a demonstrator observation for trial ``t``.

        Parameters
        ----------
        t : int
            Trial index (0-based).
        rng : numpy.random.Generator
            RNG used for stochastic demonstrator action selection and observation noise.

        Returns
        -------
        SocialObservation
            Demonstrator choice and outcome (true and observed).

        Notes
        -----
        The demonstrator updates using the **true** outcome. The returned
        ``observed_others_outcomes`` corresponds to what the subject/model sees,
        controlled by the trial's ``demo_outcome`` observation model.
        """
        rts = self.resolved_trial_spec(t=t)
        state = self.get_state()

        a = int(self.demonstrator.act(state=state, spec=self.spec, rng=rng))
        estep = self.env.step(action=a, rng=rng)
        true_outcome = float(estep.outcome)

        # Demonstrator updates using true outcome.
        self.demonstrator.update(state=state, action=a, outcome=true_outcome, spec=self.spec, rng=rng)

        observed = None
        if rts.demo_outcome is not None:
            observed = rts.demo_outcome.observe(true_outcome=true_outcome, env=self.spec, rng=rng)

        return SocialObservation(
            others_choices=[a],
            others_outcomes=[true_outcome],
            observed_others_outcomes=None if observed is None else [float(observed)],
            info={"demo_outcome_obs": _obs_spec_to_dict(rts.demo_outcome)},
        )
