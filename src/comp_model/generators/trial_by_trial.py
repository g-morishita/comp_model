from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..data.types import Trial, Block, SubjectData, StudyData
from ..interfaces.model import ComputationalModel, SocialComputationalModel
from ..interfaces.generator import Generator
from ..plans.block import BlockPlan

BanditFactory = Callable[[Mapping[str, Any]], Any]


@dataclass(slots=True)
class TrialByTrialGenerator(Generator):
    """
    Simulate StudyData by replaying:
      (optional) social observation -> model.social_update
      model.action_probs -> sample action
      bandit.step -> outcome
      model.update

    Resets model and bandit at the start of each block.
    """

    def simulate_study(
        self,
        *,
        bandit_factory: BanditFactory,
        model: ComputationalModel,
        subj_params: Mapping[str, Mapping[str, float]],
        subject_block_plans: Mapping[str, Sequence[BlockPlan]],
        rng: np.random.Generator,
    ) -> StudyData:
        subjects: list[SubjectData] = []
        task_spec = None  # set from first bandit

        for subject_id, plans in subject_block_plans.items():
            if subject_id not in subj_params:
                raise ValueError(f"Missing subj_params for {subject_id}")

            model_params = subj_params[subject_id]
            model.set_params(model_params)

            blocks: list[Block] = []
            for plan in plans:
                n_trials = int(plan.n_trials)
                bandit_cfg = dict(plan.bandit_config)
                bandit = bandit_factory(bandit_cfg)

                spec = bandit.spec
                task_spec = spec if task_spec is None else task_spec

                # reset for block
                bandit.reset(rng=rng)
                model.reset_block(spec=spec)

                trials: list[Trial] = []
                for t in range(n_trials):
                    state = bandit.get_state()

                    # observe others if both data+model support it
                    others_choices = None
                    others_outcomes = None
                    social_info = None

                    if isinstance(model, SocialComputationalModel) and getattr(spec, "is_social", False):
                        obs = bandit.observe_others(rng=rng)
                        others_choices = obs.others_choices
                        others_outcomes = obs.others_outcomes
                        social_info = obs.info

                        model.social_update(state=state, social=obs, spec=spec, info=None)

                    probs = model.action_probs(state=state, spec=spec)
                    action = int(rng.choice(spec.n_actions, p=probs))

                    outcome = float(bandit.step(action=action, rng=rng).outcome)
                    model.update(state=state, action=action, outcome=outcome, spec=spec, info=None)

                    trials.append(
                        Trial(
                            t=t,
                            state=state,
                            choice=action,
                            outcome=outcome,
                            info={},
                            others_choices=others_choices,
                            others_outcomes=others_outcomes,
                            social_info=social_info,
                        )
                    )

                blocks.append(
                    Block(
                        block_id=str(plan.get("block_id", f"block_{len(blocks)+1}")),
                        trials=trials,
                        task_spec=spec,
                        metadata={"bandit_cfg": bandit_cfg},
                    )
                )

            subjects.append(SubjectData(subject_id=subject_id, blocks=blocks, metadata={}))

        if task_spec is None:
            raise ValueError("No subjects/blocks were simulated.")

        return StudyData(subjects=subjects, task_spec=task_spec, metadata={})
