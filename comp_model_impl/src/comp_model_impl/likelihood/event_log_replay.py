from __future__ import annotations

from typing import Mapping

import numpy as np

from comp_model_core.data.types import StudyData, SubjectData
from comp_model_core.events.accessors import get_event_log
from comp_model_core.events.types import EventType
from comp_model_core.interfaces.bandit import SocialObservation
from comp_model_core.interfaces.model import ComputationalModel, SocialComputationalModel

_EPS = 1e-12


def loglike_subject(
    *,
    subject: SubjectData,
    model: ComputationalModel,
    params: Mapping[str, float],
) -> float:
    """
    Event-log replay for a single subject.

    Semantics are fully determined by the event stream. In particular, BLOCK_START
    events indicate when the model must reset.

    Notes
    -----
    This function never mutates the passed-in ``model``. It evaluates likelihood
    using a fresh ``model.clone()`` instance so it is safe under repeated calls,
    multi-start optimization, and parallelism.
    """
    m = model.clone()
    m.set_params(params)
    ll = 0.0
    is_social_model = isinstance(m, SocialComputationalModel)

    for block in subject.blocks:
        spec = block.task_spec
        if spec is None:
            raise ValueError("Block.task_spec is None; required for replay.")

        log = get_event_log(block)

        for e in log.events:
            if e.type is EventType.BLOCK_START:
                m.reset_block(spec=spec)
                continue

            if e.type is EventType.SOCIAL_OBSERVED:
                if not is_social_model:
                    continue

                p = e.payload
                social = SocialObservation(
                    others_choices=p.get("others_choices"),
                    others_outcomes=p.get("others_outcomes"),
                    observed_others_outcomes=p.get("observed_others_outcomes"),
                    info=p.get("social_info"),
                )
                m.social_update(state=e.state, social=social, spec=spec, info=None)
                continue

            if e.type is EventType.CHOICE:
                choice = p_choice = e.payload.get("choice", None)
                if p_choice is None:
                    continue
                probs = m.action_probs(state=e.state, spec=spec)
                p = float(probs[int(choice)])
                ll += float(np.log(max(p, _EPS)))
                continue

            if e.type is EventType.OUTCOME:
                action = int(e.payload["action"])
                observed_outcome = e.payload.get("observed_outcome", None)
                info = e.payload.get("info", None)
                m.update(
                    state=e.state,
                    action=action,
                    outcome=observed_outcome,
                    spec=spec,
                    info=info,
                )
                continue

            raise ValueError(f"Unknown event type: {e.type}")

    return ll


def loglike_study_independent(
    *,
    study: StudyData,
    model: ComputationalModel,
    subject_params: Mapping[str, Mapping[str, float]],
) -> float:
    ll = 0.0
    for subj in study.subjects:
        ll += loglike_subject(subject=subj, model=model, params=subject_params[subj.subject_id])
    return ll
