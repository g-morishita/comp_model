from __future__ import annotations

import numpy as np
from typing import Mapping

from comp_model_core.data.types import StudyData, SubjectData
from comp_model_core.interfaces.model import ComputationalModel, SocialComputationalModel
from comp_model_core.interfaces.bandit import SocialObservation

_EPS = 1e-12


def loglike_subject(
    *,
    subject: SubjectData,
    model: ComputationalModel,
    params: Mapping[str, float],
) -> float:
    """
    Trial replay for a single subject.

    Order per trial:
      1) (optional) social_update BEFORE choice likelihood
      2) add log p(choice)
      3) (optional) update from private outcome AFTER reward

    Assumes Block.task_spec is present (self-contained blocks).
    """
    model.set_params(params)
    ll = 0.0

    is_social_model = isinstance(model, SocialComputationalModel)

    for block in subject.blocks:
        spec = block.task_spec
        model.reset_block(spec=spec)

        for tr in block.trials:
            # social observation (if available in data + model supports it)
            if is_social_model and tr.others_choices:
                social = SocialObservation(
                    others_choices=tr.others_choices,
                    observed_others_outcomes=tr.observed_others_outcomes,
                    others_outcomes=tr.others_outcomes,
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
            if tr.outcome is not None:
                model.update(
                    state=tr.state,
                    action=int(tr.choice),
                    outcome=tr.observed_outcome,
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
            subject=subj,
            model=model,
            params=subject_params[subj.subject_id],
        )
    return ll
