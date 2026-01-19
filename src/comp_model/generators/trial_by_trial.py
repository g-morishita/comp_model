from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from ..data.types import Trial, Block, SubjectData
from ..interfaces.bandit import Bandit, SocialBandit
from ..interfaces.generator import Generator
from ..interfaces.model import ComputationalModel, SocialComputationalModel
from ..plans.block import BlockPlan
from ..errors import CompatibilityError

TaskBuilder = Callable[[BlockPlan], Bandit]


@dataclass(slots=True)
class TrialByTrialGenerator(Generator):
    """
    Trial-by-trial simulation.

    Order:
      (optional) others observation -> model.social_update
      model.action_probs -> sample action
      bandit.step -> outcome
      model.update

    Resets model and bandit at the start of each block.
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
        # subject parameters shared across blocks
        model.set_params(params)

        blocks: list[Block] = []

        for plan in block_plans:
            bandit = task_builder(plan)
            spec = bandit.spec

            if not model.supports(spec=spec):
                raise CompatibilityError("The computational model is not compatible with the current task.")

            # reset for block
            bandit.reset(rng=rng)
            model.reset_block(spec=spec)

            trials: list[Trial] = []

            for t in range(int(plan.n_trials)):
                state = bandit.get_state()

                others_choices = None
                others_outcomes = None
                social_info = None

                if (
                    isinstance(bandit, SocialBandit)
                    and isinstance(model, SocialComputationalModel)
                    and getattr(spec, "is_social", False)
                ):
                    obs = bandit.observe_others(rng=rng)
                    others_choices = obs.others_choices
                    others_outcomes = obs.others_outcomes
                    social_info = obs.info

                    model.social_update(state=state, social=obs, spec=spec, info=None)

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
                        others_choices=others_choices,
                        others_outcomes=others_outcomes,
                        social_info=social_info or {},
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
