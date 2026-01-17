from __future__ import annotations

import numpy as np
from typing import Mapping

from ..data.types import StudyData, SubjectData
from ..interfaces.model import ComputationalModel, SocialComputationalModel
from ..interfaces.bandit import SocialObservation

_EPS = 1e-12


def loglike_subject(
    *,
    study: StudyData,
    subject: SubjectData,
    model: ComputationalModel,
    params: Mapping[str, float],
) -> float:
    """
    Generic trial replay:
      - sets subject-level parameters once
      - resets latent state per block
      - applies optional social_update BEFORE choice likelihood if trial has others_choices
      - adds log p(choice_t)
      - applies update AFTER reward if reward exists

    Works for:
      - asocial models (ComputationalModel)
      - social models (SocialComputationalModel)
      - multi-block data (SubjectData.blocks)
    """
    model.set_params(params)
    ll = 0.0

    is_social_model = isinstance(model, SocialComputationalModel)

    for block in subject.blocks:
        spec = study.task_for_block(block)
        model.reset_block(spec=spec)

        for tr in block.trials:
            # social observation (if available in data + model supports it)
            if is_social_model and tr.others_choices:
                social = SocialObservation(
                    others_choices=tr.others_choices,
                    others_outcomes=tr.others_rewards,
                    info=tr.social_info,
                )
                model.social_update(state=tr.state, social=social, spec=spec, info=None)

            # skip if no choice (pure observation trial)
            if tr.choice is None:
                continue

            probs = model.action_probs(state=tr.state, spec=spec)
            p = float(probs[int(tr.choice)])
            ll += float(np.log(max(p, _EPS)))

            # update from private outcome if present
            if tr.reward is not None:
                model.update(
                    state=tr.state,
                    action=int(tr.choice),
                    reward=float(tr.reward),
                    spec=spec,
                    info=tr.info,
                )

    return ll


def loglike_study_independent(
    *,
    study: StudyData,
    model: ComputationalModel,
    subject_params: Mapping[str, Mapping[str, float]],
) -> float:
    """
    Sum log-likelihood across subjects (independent subject fits).
    """
    ll = 0.0
    for subj in study.subjects:
        ll += loglike_subject(
            study=study,
            subject=subj,
            model=model,
            params=subject_params[subj.subject_id],
        )
    return ll
