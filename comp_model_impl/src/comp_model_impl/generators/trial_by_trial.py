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

TaskBuilder = Callable[[BlockPlan], Bandit]


def _ensure_model_supports(model: ComputationalModel, bandit: Bandit) -> None:
    spec = bandit.spec
    if not model.supports(spec=spec):
        raise CompatibilityError("The computational model is not compatible with the current task.")


def _reset_block(model: ComputationalModel, bandit: Bandit, rng: np.random.Generator) -> None:
    bandit.reset(rng=rng)
    model.reset_block(spec=bandit.spec)


@dataclass(slots=True)
class AsocialBanditGenerator(Generator):
    """
    Asocial trial simulation.

    Order:
      model.action_probs -> sample action
      bandit.step -> outcome
      model.update

    This generator is strict:
      - expects an asocial task (spec.is_social must be False)
      - does not call any observation methods
      - writes social fields in Trial as None / {} (no observation exists)
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
                raise CompatibilityError(
                    "AsocialBanditGenerator cannot run a social task (spec.is_social=True). "
                    "Use a Social*Generator instead."
                )

            _ensure_model_supports(model, bandit)
            _reset_block(model, bandit, rng)

            trials: list[Trial] = []

            for t in range(int(plan.n_trials)):
                state = bandit.get_state()

                probs = model.action_probs(state=state, spec=spec)
                action = int(rng.choice(spec.n_actions, p=probs))

                step = bandit.step(action=action, rng=rng)
                outcome = float(step.outcome)

                model.update(state=state, action=action, outcome=outcome, spec=spec, info=step.info)

                trials.append(
                    Trial(
                        t=t,
                        state=state,
                        choice=action,
                        outcome=outcome,
                        info=step.info or {},
                        others_choices=None,
                        others_outcomes=None,
                        social_info={},  # asocial means no observation exists
                    )
                )

            blocks.append(
                Block(
                    block_id=plan.block_id,
                    trials=trials,
                    task_spec=spec,
                    metadata={"plan": dict(plan.metadata)},
                )
            )

        return SubjectData(subject_id=subject_id, blocks=blocks, metadata={})


@dataclass(slots=True)
class SocialPreChoiceGenerator(Generator):
    """
    Social trial simulation (observe others BEFORE choice).

    Order:
      others observation -> model.social_update
      model.action_probs -> sample action
      bandit.step -> outcome
      model.update

    Strict:
      - requires spec.is_social=True
      - requires bandit is SocialBandit
      - requires model is SocialComputationalModel
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
                raise CompatibilityError(
                    "SocialPreChoiceGenerator requires a social task (spec.is_social=True)."
                )

            if not isinstance(bandit, SocialBandit):
                raise CompatibilityError("Social task requires a SocialBandit task object.")
            if not isinstance(model, SocialComputationalModel):
                raise CompatibilityError("Social task requires a SocialComputationalModel.")

            _ensure_model_supports(model, bandit)
            _reset_block(model, bandit, rng)

            sb = cast(SocialBandit, bandit)
            sm = cast(SocialComputationalModel, model)

            trials: list[Trial] = []

            for t in range(int(plan.n_trials)):
                state = sb.get_state()

                obs = sb.observe_others(rng=rng)
                sm.social_update(state=state, social=obs, spec=spec, info=None)

                probs = sm.action_probs(state=state, spec=spec)
                action = int(rng.choice(spec.n_actions, p=probs))

                step = sb.step(action=action, rng=rng)
                outcome = float(step.outcome)

                sm.update(state=state, action=action, outcome=outcome, spec=spec, info=step.info)

                trials.append(
                    Trial(
                        t=t,
                        state=state,
                        choice=action,
                        outcome=outcome,
                        info=step.info or {},
                        others_choices=obs.others_choices,
                        others_outcomes=obs.others_outcomes,
                        social_info=obs.info or {},
                    )
                )

            blocks.append(
                Block(
                    block_id=plan.block_id,
                    trials=trials,
                    task_spec=spec,
                    metadata={"plan": dict(plan.metadata)},
                )
            )

        return SubjectData(subject_id=subject_id, blocks=blocks, metadata={})


@dataclass(slots=True)
class SocialPostOutcomeGenerator(Generator):
    """
    Social trial simulation (observe others AFTER self outcome).

    Order:
      model.action_probs -> sample action
      bandit.step -> outcome
      model.update
      others observation -> model.social_update

    Strict social requirements (same as SocialPreChoiceGenerator).
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
                raise CompatibilityError(
                    "SocialPostOutcomeGenerator requires a social task (spec.is_social=True)."
                )

            if not isinstance(bandit, SocialBandit):
                raise CompatibilityError("Social task requires a SocialBandit task object.")
            if not isinstance(model, SocialComputationalModel):
                raise CompatibilityError("Social task requires a SocialComputationalModel.")

            _ensure_model_supports(model, bandit)
            _reset_block(model, bandit, rng)

            sb = cast(SocialBandit, bandit)
            sm = cast(SocialComputationalModel, model)

            trials: list[Trial] = []

            for t in range(int(plan.n_trials)):
                state = sb.get_state()

                probs = sm.action_probs(state=state, spec=spec)
                action = int(rng.choice(spec.n_actions, p=probs))

                step = sb.step(action=action, rng=rng)
                outcome = float(step.outcome)

                sm.update(state=state, action=action, outcome=outcome, spec=spec, info=step.info)

                obs = sb.observe_others(rng=rng)
                sm.social_update(state=state, social=obs, spec=spec, info=None)

                trials.append(
                    Trial(
                        t=t,
                        state=state,
                        choice=action,
                        outcome=outcome,
                        info=step.info or {},
                        others_choices=obs.others_choices,
                        others_outcomes=obs.others_outcomes,
                        social_info=obs.info or {},
                    )
                )

            blocks.append(
                Block(
                    block_id=plan.block_id,
                    trials=trials,
                    task_spec=spec,
                    metadata={"plan": dict(plan.metadata)},
                )
            )

        return SubjectData(subject_id=subject_id, blocks=blocks, metadata={})
