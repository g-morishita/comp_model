"""comp_model_impl.generators.trial_by_trial

Trial-by-trial simulation generators.

These generators simulate blocks by interacting with a
:class:`comp_model_core.interfaces.block_runner.BlockRunner` (or SocialBlockRunner),
writing outputs into the "trial-by-trial" data containers
(:class:`comp_model_core.data.types.Trial`, :class:`comp_model_core.data.types.Block`,
:class:`comp_model_core.data.types.SubjectData`).

The main difference between the social generators is *timing*:
- :class:`SocialPreChoiceGenerator`: observe others BEFORE the subject chooses.
- :class:`SocialPostOutcomeGenerator`: observe others AFTER the subject receives outcome.

Notes
-----
- Trial-level interface constraints (available actions; outcome observation models)
  are supplied by the runner via ``runner.resolved_trial_spec(t=...)``.
- Action probability masking/renormalization is handled in this module using
  ``TrialSpec.available_actions``.

See Also
--------
comp_model_core.interfaces.generator.Generator
comp_model_core.interfaces.block_runner.BlockRunner
comp_model_core.interfaces.block_runner.SocialBlockRunner
comp_model_core.interfaces.model.ComputationalModel
comp_model_core.interfaces.model.SocialComputationalModel
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence, cast

import numpy as np

from comp_model_core.data.types import Trial, Block, SubjectData
from comp_model_core.errors import CompatibilityError
from comp_model_core.interfaces.generator import Generator
from comp_model_core.interfaces.model import ComputationalModel, SocialComputationalModel
from comp_model_core.interfaces.block_runner import BlockRunner, SocialBlockRunner
from comp_model_core.plans.block import BlockPlan
from comp_model_core.validation import validate_action_sets, validate_runner_against_model_requirements

BlockRunnerBuilder = Callable[[BlockPlan], BlockRunner]


def _ensure_model_supports(model: ComputationalModel, runner: BlockRunner) -> None:
    """Validate that the model supports the runner's environment spec.

    Parameters
    ----------
    model : ComputationalModel
        Model instance to validate.
    runner : BlockRunner
        Runner providing the environment specification.

    Raises
    ------
    CompatibilityError
        If ``model.supports(spec)`` returns False.
    """
    if not model.supports(spec=runner.spec):
        raise CompatibilityError("The computational model is not compatible with the current environment.")


def _reset_block(model: ComputationalModel, runner: BlockRunner, rng: np.random.Generator) -> None:
    """Reset environment and model for a new block.

    Parameters
    ----------
    model : ComputationalModel
        Model to reset.
    runner : BlockRunner
        Runner/environment to reset.
    rng : numpy.random.Generator
        RNG forwarded to runner reset.
    """
    runner.reset(rng=rng)
    model.reset_block(spec=runner.spec)


def _mask_and_renorm(probs: np.ndarray, available_actions: Sequence[int] | None) -> np.ndarray:
    """Mask probabilities to available actions and renormalize.

    Parameters
    ----------
    probs : numpy.ndarray
        Raw action probabilities from the model. Must be shape ``(n_actions,)``.
    available_actions : Sequence[int] or None
        If provided, only these actions are allowed. If None, no masking is applied.

    Returns
    -------
    numpy.ndarray
        Masked and renormalized probabilities. Sums to 1.

    Raises
    ------
    ValueError
        If the model assigns non-positive total probability mass (or all mass is on
        unavailable actions).
    """
    p = np.asarray(probs, dtype=float).copy()

    if available_actions is None:
        s = float(p.sum())
        if s <= 0:
            raise ValueError("Model returned non-positive probability mass.")
        return p / s

    mask = np.zeros_like(p, dtype=bool)
    for a in available_actions:
        mask[int(a)] = True

    p[~mask] = 0.0
    s = float(p.sum())
    if s <= 0:
        raise ValueError("All probability mass assigned to unavailable actions.")
    return p / s


@dataclass(slots=True)
class AsocialBanditGenerator(Generator):
    """Asocial trial simulation with trial-varying interface constraints.

    Notes
    -----
    Trial-level constraints include forced action sets and outcome observation models
    (hidden / noisy / etc.) provided by the runner.
    """

    def simulate_subject(
        self,
        *,
        subject_id: str,
        block_runner_builder: BlockRunnerBuilder,
        model: ComputationalModel,
        params: Mapping[str, float],
        block_plans: Sequence[BlockPlan],
        rng: np.random.Generator,
    ) -> SubjectData:
        """Simulate a subject for asocial blocks.

        Parameters
        ----------
        subject_id : str
            Subject identifier.
        block_runner_builder : callable
            Function that builds a runner from a :class:`~comp_model_core.plans.block.BlockPlan`.
        model : ComputationalModel
            Model used to generate choices and update from outcomes.
        params : Mapping[str, float]
            Model parameter values for this subject.
        block_plans : Sequence[BlockPlan]
            Block blueprints for this subject.
        rng : numpy.random.Generator
            RNG used for stochastic choices and environment noise.

        Returns
        -------
        SubjectData
            Simulated subject data.

        Raises
        ------
        CompatibilityError
            If a block is social or if model/runner compatibility checks fail.
        """
        model.set_params(params)
        blocks: list[Block] = []

        for plan in block_plans:
            runner = block_runner_builder(plan)
            spec = runner.spec

            if spec.is_social:
                raise CompatibilityError(
                    "AsocialBanditGenerator cannot run a social task (spec.is_social=True). "
                    "Use a Social*Generator instead."
                )

            _ensure_model_supports(model, runner)
            validate_action_sets(runner=runner, n_trials=int(plan.n_trials), block_id=plan.block_id)
            validate_runner_against_model_requirements(
                runner=runner,
                n_trials=int(plan.n_trials),
                reqs=model.requirements(),
                block_id=plan.block_id,
            )
            _reset_block(model, runner, rng)

            trials: list[Trial] = []

            for t in range(int(plan.n_trials)):
                state = runner.get_state()
                rts = runner.resolved_trial_spec(t=t)
                aa = rts.available_actions  # None means unconstrained

                probs = model.action_probs(state=state, spec=spec)
                probs = _mask_and_renorm(probs, aa)

                action = int(rng.choice(int(spec.n_actions), p=probs))
                step = runner.step(t=t, action=action, rng=rng)

                model.update(
                    state=state,
                    action=action,
                    outcome=step.observed_outcome,
                    spec=spec,
                    info=step.info,
                    rng=rng,
                )

                trials.append(
                    Trial(
                        t=t,
                        state=state,
                        choice=action,
                        observed_outcome=step.observed_outcome,
                        outcome=step.outcome,
                        available_actions=None if aa is None else list(aa),
                        info=step.info or {},
                        others_choices=None,
                        others_outcomes=None,
                        observed_others_outcomes=None,
                        social_info={},
                    )
                )

            blocks.append(
                Block(
                    block_id=plan.block_id,
                    trials=trials,
                    env_spec=spec,
                    metadata={"plan": dict(plan.metadata)},
                )
            )

        return SubjectData(subject_id=subject_id, blocks=blocks, metadata={})


@dataclass(slots=True)
class SocialPreChoiceGenerator(Generator):
    """Social simulation where social observation occurs BEFORE subject choice."""

    def simulate_subject(
        self,
        *,
        subject_id: str,
        block_runner_builder: BlockRunnerBuilder,
        model: ComputationalModel,
        params: Mapping[str, float],
        block_plans: Sequence[BlockPlan],
        rng: np.random.Generator,
    ) -> SubjectData:
        """Simulate a subject for social blocks (pre-choice social timing)."""
        model.set_params(params)
        blocks: list[Block] = []

        for plan in block_plans:
            runner = block_runner_builder(plan)
            spec = runner.spec

            if not spec.is_social:
                raise CompatibilityError("SocialPreChoiceGenerator requires spec.is_social=True.")
            if not isinstance(runner, SocialBlockRunner):
                raise CompatibilityError("Social blocks require a SocialBlockRunner runtime object.")
            if not isinstance(model, SocialComputationalModel):
                raise CompatibilityError("Social task requires a SocialComputationalModel.")

            _ensure_model_supports(model, runner)
            validate_action_sets(runner=runner, n_trials=int(plan.n_trials), block_id=plan.block_id)
            validate_runner_against_model_requirements(
                runner=runner,
                n_trials=int(plan.n_trials),
                reqs=model.requirements(),
                block_id=plan.block_id,
            )
            _reset_block(model, runner, rng)

            stask = cast(SocialBlockRunner, runner)
            sm = cast(SocialComputationalModel, model)

            trials: list[Trial] = []

            for t in range(int(plan.n_trials)):
                state = stask.get_state()

                obs = stask.observe_others(t=t, rng=rng)
                sm.social_update(state=state, social=obs, spec=spec, info=None, rng=rng)

                rts = stask.resolved_trial_spec(t=t)
                aa = rts.available_actions

                probs = sm.action_probs(state=state, spec=spec)
                probs = _mask_and_renorm(probs, aa)

                action = int(rng.choice(int(spec.n_actions), p=probs))
                step = stask.step(t=t, action=action, rng=rng)

                sm.update(state=state, action=action, outcome=step.observed_outcome, spec=spec, info=step.info, rng=rng)

                trials.append(
                    Trial(
                        t=t,
                        state=state,
                        choice=action,
                        observed_outcome=step.observed_outcome,
                        outcome=step.outcome,
                        available_actions=None if aa is None else list(aa),
                        info=step.info or {},
                        others_choices=obs.others_choices,
                        others_outcomes=obs.others_outcomes,
                        observed_others_outcomes=obs.observed_others_outcomes,
                        social_info=obs.info or {},
                    )
                )

            blocks.append(Block(block_id=plan.block_id, trials=trials, env_spec=spec, metadata={"plan": dict(plan.metadata)}))

        return SubjectData(subject_id=subject_id, blocks=blocks, metadata={})


@dataclass(slots=True)
class SocialPostOutcomeGenerator(Generator):
    """Social simulation where social observation occurs AFTER subject outcome."""

    def simulate_subject(
        self,
        *,
        subject_id: str,
        block_runner_builder: BlockRunnerBuilder,
        model: ComputationalModel,
        params: Mapping[str, float],
        block_plans: Sequence[BlockPlan],
        rng: np.random.Generator,
    ) -> SubjectData:
        """Simulate a subject for social blocks (post-outcome social timing)."""
        model.set_params(params)
        blocks: list[Block] = []

        for plan in block_plans:
            runner = block_runner_builder(plan)
            spec = runner.spec

            if not spec.is_social:
                raise CompatibilityError("SocialPostOutcomeGenerator requires spec.is_social=True.")
            if not isinstance(runner, SocialBlockRunner):
                raise CompatibilityError("Social blocks require a SocialBlockRunner runtime object.")
            if not isinstance(model, SocialComputationalModel):
                raise CompatibilityError("Social task requires a SocialComputationalModel.")

            _ensure_model_supports(model, runner)
            validate_action_sets(runner=runner, n_trials=int(plan.n_trials), block_id=plan.block_id)
            validate_runner_against_model_requirements(
                runner=runner,
                n_trials=int(plan.n_trials),
                reqs=model.requirements(),
                block_id=plan.block_id,
            )
            _reset_block(model, runner, rng)

            stask = cast(SocialBlockRunner, runner)
            sm = cast(SocialComputationalModel, model)

            trials: list[Trial] = []

            for t in range(int(plan.n_trials)):
                state = stask.get_state()
                rts = stask.resolved_trial_spec(t=t)
                aa = rts.available_actions

                probs = sm.action_probs(state=state, spec=spec)
                probs = _mask_and_renorm(probs, aa)

                action = int(rng.choice(int(spec.n_actions), p=probs))
                step = stask.step(t=t, action=action, rng=rng)
                sm.update(state=state, action=action, outcome=step.observed_outcome, spec=spec, info=step.info, rng=rng)

                obs = stask.observe_others(t=t, rng=rng)
                sm.social_update(state=state, social=obs, spec=spec, info=None, rng=rng)

                trials.append(
                    Trial(
                        t=t,
                        state=state,
                        choice=action,
                        observed_outcome=step.observed_outcome,
                        outcome=step.outcome,
                        available_actions=None if aa is None else list(aa),
                        info=step.info or {},
                        others_choices=obs.others_choices,
                        others_outcomes=obs.others_outcomes,
                        observed_others_outcomes=obs.observed_others_outcomes,
                        social_info=obs.info or {},
                    )
                )

            blocks.append(Block(block_id=plan.block_id, trials=trials, env_spec=spec, metadata={"plan": dict(plan.metadata)}))

        return SubjectData(subject_id=subject_id, blocks=blocks, metadata={})
