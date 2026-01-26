from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence, cast

import numpy as np

from comp_model_core.data.types import Trial, Block, SubjectData
from comp_model_core.errors import CompatibilityError
from comp_model_core.interfaces.bandit import BanditEnv, SocialBanditEnv
from comp_model_core.interfaces.generator import Generator
from comp_model_core.interfaces.block_runner import BlockRunner, SocialBlockRunner
from comp_model_core.interfaces.model import ComputationalModel, SocialComputationalModel
from comp_model_core.plans.block import BlockPlan
from comp_model_core.events.types import Event, EventLog, EventType
from comp_model_core.validation import validate_block_plan

BlockRunnerBuilder = Callable[[BlockPlan], BlockRunner]


def _ensure_model_supports(model: ComputationalModel, bandit: BanditEnv) -> None:
    if not model.supports(spec=bandit.spec):
        raise CompatibilityError("The computational model is not compatible with the current task.")


def _reset_block(model: ComputationalModel, bandit: BanditEnv, rng: np.random.Generator) -> None:
    bandit.reset(rng=rng)
    model.reset_block(spec=bandit.spec)


def _build_event_log(events: list[Event], *, metadata: dict) -> EventLog:
    return EventLog(events=events, metadata=metadata)


def _mask_and_renorm(probs: np.ndarray, available_actions: Sequence[int] | None) -> np.ndarray:
    p = np.asarray(probs, dtype=float).copy()
    if available_actions is None:
        s = float(p.sum())
        if s <= 0:
            raise ValueError("Model returned non-positive probability mass.")
        return p / s


@dataclass(slots=True)
class EventLogAsocialGenerator(Generator):
    """
    Asocial simulation with explicit event log.

    Per block:
      BLOCK_START (signals reset)
      For each trial:
        CHOICE -> OUTCOME

    The event log is stored in Block.event_log.
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
        model.set_params(params)

        blocks: list[Block] = []

        for plan in block_plans:
            runner = block_runner_builder(plan)
            spec = runner.spec

            if getattr(spec, "is_social", False):
                raise CompatibilityError("EventLogAsocialGenerator cannot run a social task (spec.is_social=True).")

            _ensure_model_supports(model, runner)
            validate_block_plan(plan=plan, env_spec=spec, requirements=model.requirements())
            _reset_block(model, runner, rng)

            trials: list[Trial] = []
            events: list[Event] = []

            # Mandatory: block reset marker
            events.append(Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={"block_id": plan.block_id}))
            idx = 1

            for t in range(int(plan.n_trials)):
                state = runner.get_state()
                ts = runner.trial_spec(t=t)
                aa = ts.available_actions

                probs = model.action_probs(state=state, spec=spec)
                probs = _mask_and_renorm(probs, aa)

                action = int(rng.choice(spec.n_actions, p=probs))
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

                model.update(state=state, action=action, outcome=step.observed_outcome, spec=spec, info=step.info)

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
                    event_log=_build_event_log(events, metadata={"timing": "asocial"}),
                    metadata={
                        "plan": dict(plan.metadata),
                    },
                )
            )

        return SubjectData(subject_id=subject_id, blocks=blocks, metadata={})


@dataclass(slots=True)
class EventLogSocialPreChoiceGenerator(Generator):
    """
    Social simulation with event log: SOCIAL_OBSERVED happens BEFORE CHOICE.

    Per block:
      BLOCK_START
      For each trial:
        SOCIAL_OBSERVED -> CHOICE -> OUTCOME
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
            validate_block_plan(plan=plan, env_spec=spec, requirements=model.requirements())
            _reset_block(model, runner, rng)

            sb = cast(SocialBlockRunner, runner)
            sm = cast(SocialComputationalModel, model)

            trials: list[Trial] = []
            events: list[Event] = []

            events.append(Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={"block_id": plan.block_id}))
            idx = 1

            for t in range(int(plan.n_trials)):
                state = sb.get_state()

                obs = sb.observe_others(t=t, rng=rng)
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
                sm.social_update(state=state, social=obs, spec=spec, info=None)

                ts = sb.trial_spec(t=t)
                aa = ts.available_actions

                probs = sm.action_probs(state=state, spec=spec)
                probs = _mask_and_renorm(probs, aa)

                action = int(rng.choice(spec.n_actions, p=probs))
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

                step = sb.step(t=t, action=action, rng=rng)
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

                sm.update(state=state, action=action, outcome=step.observed_outcome, spec=spec, info=step.info)

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
                    task_spec=spec,
                    event_log=_build_event_log(events, metadata={"timing": "pre_choice"}),
                    metadata={
                        "plan": dict(plan.metadata),
                    },
                )
            )

        return SubjectData(subject_id=subject_id, blocks=blocks, metadata={})


@dataclass(slots=True)
class EventLogSocialPostOutcomeGenerator(Generator):
    """
    Social simulation with event log: SOCIAL_OBSERVED happens AFTER OUTCOME.

    Per block:
      BLOCK_START
      For each trial:
        CHOICE -> OUTCOME -> SOCIAL_OBSERVED
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
            validate_block_plan(plan=plan, env_spec=spec, requirements=model.requirements())
            _reset_block(model, runner, rng)

            sb = cast(SocialBlockRunner, runner)
            sm = cast(SocialComputationalModel, model)

            trials: list[Trial] = []
            events: list[Event] = []

            events.append(Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={"block_id": plan.block_id}))
            idx = 1

            for t in range(int(plan.n_trials)):
                state = sb.get_state()
                ts = sb.trial_spec(t=t)
                aa = ts.available_actions

                probs = sm.action_probs(state=state, spec=spec)
                probs = _mask_and_renorm(probs, aa)

                action = int(rng.choice(spec.n_actions, p=probs))
                events.append(Event(idx=idx, type=EventType.CHOICE, t=t, state=state, payload={"choice": action, "available_actions": None if aa is None else list(aa)}))
                idx += 1

                step = sb.step(t=t, action=action, rng=rng)
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

                sm.update(state=state, action=action, outcome=step.observed_outcome, spec=spec, info=step.info)

                obs = sb.observe_others(t=t, rng=rng)
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

                sm.social_update(state=state, social=obs, spec=spec, info=None)

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
                    task_spec=spec,
                    event_log=_build_event_log(events, metadata={"timing": "post_outcome"}),
                    metadata={
                        "plan": dict(plan.metadata),
                    },
                )
            )

        return SubjectData(subject_id=subject_id, blocks=blocks, metadata={})
