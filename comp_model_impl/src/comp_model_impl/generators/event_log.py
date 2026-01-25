"""comp_model_impl.generators.event_log

Event-log simulation generators.

These generators simulate blocks like the trial-by-trial generators, but also emit
a detailed **event log** into block metadata (under ``EVENT_LOG_KEY``). The event log
captures a time-ordered sequence of events such as:

- BLOCK_START
- SOCIAL_OBSERVED (for social tasks)
- CHOICE
- OUTCOME

This format is useful for replay likelihoods that require a precise ordering of
observations and updates.

See Also
--------
comp_model_core.events.types.Event
comp_model_core.events.types.EventLog
comp_model_core.events.types.EventType
comp_model_core.events.types.EVENT_LOG_KEY
comp_model_impl.generators.trial_by_trial
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
from comp_model_core.events.types import EVENT_LOG_KEY, Event, EventLog, EventType
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


def _events_to_json(events: list[Event], *, metadata: dict) -> dict:
    """Serialize an event list into JSON payload.

    Parameters
    ----------
    events : list of Event
        Events in chronological order.
    metadata : dict
        Metadata stored in the event log object.

    Returns
    -------
    dict
        JSON-serializable event log.
    """
    return EventLog(events=events, metadata=metadata).to_json()


@dataclass(slots=True)
class EventLogAsocialGenerator(Generator):
    """Asocial simulation producing an explicit event log."""

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
        """Simulate a subject and attach an event log to each block."""
        model.set_params(params)
        blocks: list[Block] = []

        for plan in block_plans:
            runner = block_runner_builder(plan)
            spec = runner.spec

            if spec.is_social:
                raise CompatibilityError("EventLogAsocialGenerator cannot run a social task (spec.is_social=True).")

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
            events: list[Event] = []

            events.append(Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={"block_id": plan.block_id}))
            idx = 1

            for t in range(int(plan.n_trials)):
                state = runner.get_state()
                rts = runner.resolved_trial_spec(t=t)
                aa = rts.available_actions

                probs = model.action_probs(state=state, spec=spec)
                probs = _mask_and_renorm(probs, aa)

                action = int(rng.choice(int(spec.n_actions), p=probs))
                events.append(
                    Event(
                        idx=idx,
                        type=EventType.CHOICE,
                        t=t,
                        state=state,
                        payload={
                            "choice": action,
                            "available_actions": None if aa is None else list(aa),
                        },
                    )
                )
                idx += 1

                step = runner.step(t=t, action=action, rng=rng)
                events.append(
                    Event(
                        idx=idx,
                        type=EventType.OUTCOME,
                        t=t,
                        state=state,
                        payload={
                            "action": action,
                            "observed_outcome": step.observed_outcome,
                            "outcome": step.outcome,
                            "info": step.info or {},
                        },
                    )
                )
                idx += 1

                model.update(state=state, action=action, outcome=step.observed_outcome, spec=spec, info=step.info, rng=rng)

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
                    metadata={
                        "plan": dict(plan.metadata),
                        EVENT_LOG_KEY: _events_to_json(events, metadata={"timing": "asocial"}),
                    },
                )
            )

        return SubjectData(subject_id=subject_id, blocks=blocks, metadata={})


@dataclass(slots=True)
class EventLogSocialPreChoiceGenerator(Generator):
    """Social simulation with event log where SOCIAL_OBSERVED occurs BEFORE CHOICE."""

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
        """Simulate social blocks with pre-choice social timing and event log."""
        model.set_params(params)
        blocks: list[Block] = []

        for plan in block_plans:
            runner = block_runner_builder(plan)
            spec = runner.spec

            if not spec.is_social:
                raise CompatibilityError("EventLogSocialPreChoiceGenerator requires spec.is_social=True.")
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
            events: list[Event] = []

            events.append(Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={"block_id": plan.block_id}))
            idx = 1

            for t in range(int(plan.n_trials)):
                state = stask.get_state()

                obs = stask.observe_others(t=t, rng=rng)
                events.append(
                    Event(
                        idx=idx,
                        type=EventType.SOCIAL_OBSERVED,
                        t=t,
                        state=state,
                        payload={
                            "others_choices": list(obs.others_choices or []),
                            "others_outcomes": list(obs.others_outcomes or []),
                            "observed_others_outcomes": None if obs.observed_others_outcomes is None else list(obs.observed_others_outcomes),
                            "social_info": obs.info or {},
                        },
                    )
                )
                idx += 1
                sm.social_update(state=state, social=obs, spec=spec, info=None, rng=rng)

                rts = stask.resolved_trial_spec(t=t)
                aa = rts.available_actions

                probs = sm.action_probs(state=state, spec=spec)
                probs = _mask_and_renorm(probs, aa)

                action = int(rng.choice(int(spec.n_actions), p=probs))
                events.append(
                    Event(
                        idx=idx,
                        type=EventType.CHOICE,
                        t=t,
                        state=state,
                        payload={"choice": action, "available_actions": None if aa is None else list(aa)},
                    )
                )
                idx += 1

                step = stask.step(t=t, action=action, rng=rng)
                events.append(
                    Event(
                        idx=idx,
                        type=EventType.OUTCOME,
                        t=t,
                        state=state,
                        payload={"action": action, "observed_outcome": step.observed_outcome, "outcome": step.outcome, "info": step.info or {}},
                    )
                )
                idx += 1

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

            blocks.append(
                Block(
                    block_id=plan.block_id,
                    trials=trials,
                    env_spec=spec,
                    metadata={"plan": dict(plan.metadata), EVENT_LOG_KEY: _events_to_json(events, metadata={"timing": "pre_choice"})},
                )
            )

        return SubjectData(subject_id=subject_id, blocks=blocks, metadata={})


@dataclass(slots=True)
class EventLogSocialPostOutcomeGenerator(Generator):
    """Social simulation with event log where SOCIAL_OBSERVED occurs AFTER OUTCOME."""

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
        """Simulate social blocks with post-outcome social timing and event log."""
        model.set_params(params)
        blocks: list[Block] = []

        for plan in block_plans:
            runner = block_runner_builder(plan)
            spec = runner.spec

            if not spec.is_social:
                raise CompatibilityError("EventLogSocialPostOutcomeGenerator requires spec.is_social=True.")
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
            events: list[Event] = []

            events.append(Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={"block_id": plan.block_id}))
            idx = 1

            for t in range(int(plan.n_trials)):
                state = stask.get_state()
                rts = stask.resolved_trial_spec(t=t)
                aa = rts.available_actions

                probs = sm.action_probs(state=state, spec=spec)
                probs = _mask_and_renorm(probs, aa)

                action = int(rng.choice(int(spec.n_actions), p=probs))
                events.append(Event(idx=idx, type=EventType.CHOICE, t=t, state=state, payload={"choice": action, "available_actions": None if aa is None else list(aa)}))
                idx += 1

                step = stask.step(t=t, action=action, rng=rng)
                events.append(Event(idx=idx, type=EventType.OUTCOME, t=t, state=state, payload={"action": action, "observed_outcome": step.observed_outcome, "outcome": step.outcome, "info": step.info or {}}))
                idx += 1

                sm.update(state=state, action=action, outcome=step.observed_outcome, spec=spec, info=step.info, rng=rng)

                obs = stask.observe_others(t=t, rng=rng)
                events.append(
                    Event(
                        idx=idx,
                        type=EventType.SOCIAL_OBSERVED,
                        t=t,
                        state=state,
                        payload={
                            "others_choices": list(obs.others_choices or []),
                            "others_outcomes": list(obs.others_outcomes or []),
                            "observed_others_outcomes": None if obs.observed_others_outcomes is None else list(obs.observed_others_outcomes),
                            "social_info": obs.info or {},
                        },
                    )
                )
                idx += 1
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

            blocks.append(
                Block(
                    block_id=plan.block_id,
                    trials=trials,
                    env_spec=spec,
                    metadata={"plan": dict(plan.metadata), EVENT_LOG_KEY: _events_to_json(events, metadata={"timing": "post_outcome"})},
                )
            )

        return SubjectData(subject_id=subject_id, blocks=blocks, metadata={})
