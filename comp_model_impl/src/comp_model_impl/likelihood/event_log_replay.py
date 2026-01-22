from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from comp_model_core.data.types import StudyData, SubjectData, Block
from comp_model_core.events.types import EVENT_LOG_KEY, EventLog, EventType, validate_event_log
from comp_model_core.interfaces.bandit import SocialObservation
from comp_model_core.interfaces.model import ComputationalModel, SocialComputationalModel

_EPS = 1e-12


def _get_block_event_log(block: Block) -> EventLog:
    md = block.metadata or {}
    if EVENT_LOG_KEY not in md:
        raise ValueError(
            f"Block {block.block_id!r} is missing metadata[{EVENT_LOG_KEY!r}]. "
            "Use an EventLog*Generator (or add an event log exporter)."
        )
    raw = md[EVENT_LOG_KEY]
    if isinstance(raw, EventLog):
        log = raw
    elif isinstance(raw, dict):
        log = EventLog.from_json(raw)
    else:
        raise TypeError(f"metadata[{EVENT_LOG_KEY!r}] must be dict or EventLog, got {type(raw)}")

    validate_event_log(log)
    return log


def loglike_subject(
    *,
    subject: SubjectData,
    model: ComputationalModel,
    params: Mapping[str, float],
) -> float:
    """
    Event-log replay for a single subject.

    Semantics are fully determined by the event stream.
    In particular, BLOCK_START events indicate when the model must reset.
    """
    model.set_params(params)
    ll = 0.0
    is_social_model = isinstance(model, SocialComputationalModel)

    for block in subject.blocks:
        spec = block.task_spec
        if spec is None:
            raise ValueError("Block.task_spec is None; required for replay.")

        log = _get_block_event_log(block)

        for e in log.events:
            if e.type is EventType.BLOCK_START:
                model.reset_block(spec=spec)
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
                model.social_update(state=e.state, social=social, spec=spec, info=None)
                continue

            if e.type is EventType.CHOICE:
                choice = e.payload.get("choice", None)
                if choice is None:
                    continue
                probs = model.action_probs(state=e.state, spec=spec)
                p = float(probs[int(choice)])
                ll += float(np.log(max(p, _EPS)))
                continue

            if e.type is EventType.OUTCOME:
                action = int(e.payload["action"])
                observed_outcome = e.payload.get("observed_outcome", None)
                info = e.payload.get("info", None)
                model.update(
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
