"""Event-log replay likelihood.

This module computes log-likelihood by replaying the event stream and querying
model action probabilities at each choice event.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import copy

from comp_model_core.data.types import StudyData, SubjectData
from comp_model_core.events.accessors import get_event_log
from comp_model_core.events.types import EventType
from comp_model_core.interfaces.bandit import SocialObservation
from comp_model_core.interfaces.model import ComputationalModel, SocialComputationalModel

_EPS = 1e-12


def _mask_and_renorm(probs: np.ndarray, available_actions: Sequence[int] | None) -> np.ndarray:
    p = np.asarray(probs, dtype=float).copy()
    if available_actions is None:
        return p

    mask = np.zeros_like(p, dtype=bool)
    for a in available_actions:
        mask[int(a)] = True
    p[~mask] = 0.0
    s = float(p.sum())
    return p / max(s, _EPS)


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
    m = copy.deepcopy(model)
    m.set_params(params)
    ll = 0.0
    is_social_model = isinstance(m, SocialComputationalModel)

    for block in subject.blocks:
        spec = block.env_spec
        if spec is None:
            raise ValueError("Block.env_spec is None; required for replay.")

        log = get_event_log(block)

        for e in log.events:
            if e.type is EventType.BLOCK_START:
                # Within-subject designs: condition is block-level and must be
                # explicitly present in the event stream.
                cond = e.payload.get("condition", None)
                if cond is None:
                    raise ValueError(
                        "BLOCK_START event is missing required payload key 'condition'. "
                        "Re-generate logs with an updated generator or attach condition during ingestion."
                    )
                setter = getattr(m, "set_condition", None)
                if callable(setter):
                    setter(str(cond))

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

                aa = e.payload.get("available_actions", None)

                probs = m.action_probs(state=e.state, spec=spec)
                probs = _mask_and_renorm(probs, aa)
                
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

    """
    Compute the study log-likelihood as a sum of independent subjects.
    
    Parameters
    ----------
    study : comp_model_core.data.types.StudyData
        Study containing subjects with event logs.
    model : comp_model_core.interfaces.model.ComputationalModel
        Model instance (will be reset/replayed).
    params_by_subject : Mapping[str, Mapping[str, float]]
        Mapping from ``subject_id`` to fitted parameter dict.
    
    Returns
    -------
    float
        Total log-likelihood across all subjects.
    """

    ll = 0.0
    for subj in study.subjects:
        ll += loglike_subject(subject=subj, model=model, params=subject_params[subj.subject_id])
    return ll
