from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence, cast

import numpy as np

from comp_model_core.data.types import Trial, Block, SubjectData
from comp_model_core.errors import CompatibilityError
from comp_model_core.interfaces.bandit import Bandit, SocialBandit
from comp_model_core.interfaces.generator import Generator
from comp_model_core.interfaces.model import ComputationalModel, SocialComputationalModel
from comp_model_core.plans.block import BlockPlan
from comp_model_core.events.types import EVENT_LOG_KEY, Event, EventLog, EventType

TaskBuilder = Callable[[BlockPlan], Bandit]


def _ensure_model_supports(model: ComputationalModel, bandit: Bandit) -> None:
    if not model.supports(spec=bandit.spec):
        raise CompatibilityError("The computational model is not compatible with the current task.")


def _reset_block(model: ComputationalModel, bandit: Bandit, rng: np.random.Generator) -> None:
    bandit.reset(rng=rng)
    model.reset_block(spec=bandit.spec)


def _events_to_json(events: list[Event], *, metadata: dict) -> dict:
    return EventLog(events=events, metadata=metadata).to_json()


@dataclass(slots=True)
class EventLogAsocialGenerator(Generator):
    """
    Asocial simulation with explicit event log.

    Per block:
      BLOCK_START (signals reset)
      For each trial:
        CHOICE -> OUTCOME

    The event log is stored in Block.metadata[EVENT_LOG_KEY].
    """

    def simulate_subject(
        self,
        *,
        subject_id: str,
        task_builder: TaskBuilder,
        model: ComputationalModel,
        params: Mapping[str, float],
        block_plans: Sequence[BlockPlan],
        rng: np.random.Generator,
    ) -> SubjectData:
        model.set_params(params)

        blocks: list[Block] = []

        for plan in block_plans:
            bandit = task_builder(plan)
            spec = bandit.spec

            if getattr(spec, "is_social", False):
                raise CompatibilityError("EventLogAsocialGenerator cannot run a social task (spec.is_social=True).")

            _ensure_model_supports(model, bandit)
            _reset_block(model, bandit, rng)

            trials: list[Trial] = []
            events: list[Event] = []

            # Mandatory: block reset marker
            events.append(Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={"block_id": plan.block_id}))
            idx = 1

            for t in range(int(plan.n_trials)):
                state = bandit.get_state()

                probs = model.action_probs(state=state, spec=spec)
                action = int(rng.choice(spec.max_n_actions, p=probs))

                events.append(Event(idx=idx, type=EventType.CHOICE, t=t, state=state, payload={"choice": action}))
                idx += 1

                step = bandit.step(action=action, rng=rng)

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
                    task_spec=spec,
                    metadata={
                        "plan": dict(plan.metadata),
                        EVENT_LOG_KEY: _events_to_json(events, metadata={"timing": "asocial"}),
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
        task_builder: TaskBuilder,
        model: ComputationalModel,
        params: Mapping[str, float],
        block_plans: Sequence[BlockPlan],
        rng: np.random.Generator,
    ) -> SubjectData:
        model.set_params(params)

        blocks: list[Block] = []

        for plan in block_plans:
            bandit = task_builder(plan)
            spec = bandit.spec

            if not getattr(spec, "is_social", False):
                raise CompatibilityError("EventLogSocialPreChoiceGenerator requires a social task (spec.is_social=True).")

            if not isinstance(bandit, SocialBandit):
                raise CompatibilityError("Social task requires a SocialBandit task object.")
            if not isinstance(model, SocialComputationalModel):
                raise CompatibilityError("Social task requires a SocialComputationalModel.")

            _ensure_model_supports(model, bandit)
            _reset_block(model, bandit, rng)

            sb = cast(SocialBandit, bandit)
            sm = cast(SocialComputationalModel, model)

            trials: list[Trial] = []
            events: list[Event] = []

            events.append(Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={"block_id": plan.block_id}))
            idx = 1

            for t in range(int(plan.n_trials)):
                state = sb.get_state()

                obs = sb.observe_others(rng=rng)
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

                probs = sm.action_probs(state=state, spec=spec)
                action = int(rng.choice(spec.max_n_actions, p=probs))
                events.append(Event(idx=idx, type=EventType.CHOICE, t=t, state=state, payload={"choice": action}))
                idx += 1

                step = sb.step(action=action, rng=rng)
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
                    metadata={
                        "plan": dict(plan.metadata),
                        EVENT_LOG_KEY: _events_to_json(events, metadata={"timing": "pre_choice"}),
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
        task_builder: TaskBuilder,
        model: ComputationalModel,
        params: Mapping[str, float],
        block_plans: Sequence[BlockPlan],
        rng: np.random.Generator,
    ) -> SubjectData:
        model.set_params(params)

        blocks: list[Block] = []

        for plan in block_plans:
            bandit = task_builder(plan)
            spec = bandit.spec

            if not getattr(spec, "is_social", False):
                raise CompatibilityError("EventLogSocialPostOutcomeGenerator requires a social task (spec.is_social=True).")

            if not isinstance(bandit, SocialBandit):
                raise CompatibilityError("Social task requires a SocialBandit task object.")
            if not isinstance(model, SocialComputationalModel):
                raise CompatibilityError("Social task requires a SocialComputationalModel.")

            _ensure_model_supports(model, bandit)
            _reset_block(model, bandit, rng)

            sb = cast(SocialBandit, bandit)
            sm = cast(SocialComputationalModel, model)

            trials: list[Trial] = []
            events: list[Event] = []

            events.append(Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={"block_id": plan.block_id}))
            idx = 1

            for t in range(int(plan.n_trials)):
                state = sb.get_state()

                probs = sm.action_probs(state=state, spec=spec)
                action = int(rng.choice(spec.max_n_actions, p=probs))
                events.append(Event(idx=idx, type=EventType.CHOICE, t=t, state=state, payload={"choice": action}))
                idx += 1

                step = sb.step(action=action, rng=rng)
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

                obs = sb.observe_others(rng=rng)
                events.append(
                    Event(
                        idx=idx,
                        type=EventType.SOCIAL_OBSERVED,
                        t=t,
                        state=state,  # keep same trial's state semantics as your current code
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
                    metadata={
                        "plan": dict(plan.metadata),
                        EVENT_LOG_KEY: _events_to_json(events, metadata={"timing": "post_outcome"}),
                    },
                )
            )

        return SubjectData(subject_id=subject_id, blocks=blocks, metadata={})
