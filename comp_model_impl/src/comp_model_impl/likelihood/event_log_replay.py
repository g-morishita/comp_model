"""comp_model_impl.likelihood.event_log_replay

Replay likelihood for **event-log** datasets.

This module computes log-likelihood by replaying a recorded event log through a model.
Compared to trial-by-trial replay, the event log encodes the exact ordering of:
- social observations,
- choices,
- outcomes,
which can matter for models where updates depend on timing.

Workflow
--------
For each block:
1. Read `env_spec` from the block (required).
2. Reset the model for the block.
3. Iterate through events in chronological order:
   - SOCIAL_OBSERVED -> `model.social_update(...)` (if model supports it)
   - CHOICE          -> accumulate log-prob of the observed choice
   - OUTCOME         -> `model.update(...)` with observed outcome (may be None)

See Also
--------
comp_model_core.events.types.EventLog
comp_model_core.events.types.EventType
comp_model_impl.generators.event_log
comp_model_impl.likelihood.replay
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from comp_model_core.data.types import StudyData, SubjectData
from comp_model_core.events.types import EVENT_LOG_KEY, EventLog, EventType
from comp_model_core.interfaces.block_runner import SocialObservation
from comp_model_core.interfaces.model import ComputationalModel, SocialComputationalModel

_EPS = 1e-12


def _mask_and_renorm(probs: np.ndarray, available_actions: Sequence[int] | None) -> np.ndarray:
    """Mask probabilities to available actions and renormalize (numerically stable).

    Parameters
    ----------
    probs : numpy.ndarray
        Raw action probabilities from the model. Shape ``(n_actions,)``.
    available_actions : Sequence[int] or None
        If provided, only these actions are allowed. If None, no masking is applied.

    Returns
    -------
    numpy.ndarray
        Masked and renormalized probabilities.
    """
    p = np.asarray(probs, dtype=float).copy()

    if available_actions is None:
        s = float(p.sum())
        return p / max(s, _EPS)

    mask = np.zeros_like(p, dtype=bool)
    for a in available_actions:
        mask[int(a)] = True
    p[~mask] = 0.0
    s = float(p.sum())
    return p / max(s, _EPS)


def _load_event_log(block_metadata: Mapping[str, Any]) -> EventLog:
    """Load an EventLog object from block metadata.

    Parameters
    ----------
    block_metadata : Mapping[str, Any]
        Block metadata dictionary expected to contain EVENT_LOG_KEY.

    Returns
    -------
    EventLog
        Parsed event log.

    Raises
    ------
    KeyError
        If EVENT_LOG_KEY is missing from metadata.
    ValueError
        If parsing fails.
    """
    payload = block_metadata[EVENT_LOG_KEY]
    return EventLog.from_json(payload)


def loglike_subject_event_log(
    *,
    subject: SubjectData,
    model: ComputationalModel,
    params: Mapping[str, float],
) -> float:
    """Compute replay log-likelihood for one subject using event logs.

    Parameters
    ----------
    subject : SubjectData
        Subject dataset containing blocks with event logs in metadata.
    model : ComputationalModel
        Model instance used to compute probabilities and updates.
    params : Mapping[str, float]
        Model parameters for this subject.

    Returns
    -------
    float
        Total log-likelihood across all choice events.

    Raises
    ------
    ValueError
        If a block is missing `env_spec`.
    KeyError
        If a block is missing the event log metadata key.
    """
    model.set_params(params)
    ll = 0.0

    is_social_model = isinstance(model, SocialComputationalModel)

    for block in subject.blocks:
        spec = block.env_spec
        if spec is None:
            raise ValueError("Block.env_spec is None; required for replay.")

        model.reset_block(spec=spec)

        elog = _load_event_log(block.metadata)

        for ev in elog.events:
            if ev.type is EventType.SOCIAL_OBSERVED and is_social_model:
                payload = ev.payload or {}
                social = SocialObservation(
                    others_choices=payload.get("others_choices", None),
                    others_outcomes=payload.get("others_outcomes", None),
                    observed_others_outcomes=payload.get("observed_others_outcomes", None),
                    info=payload.get("social_info", None),
                )
                # ev.state can be None depending on logging; fall back to payload or None.
                model.social_update(state=ev.state, social=social, spec=spec, info=None)
                continue

            if ev.type is EventType.CHOICE:
                payload = ev.payload or {}
                choice = payload.get("choice", None)
                if choice is None:
                    continue
                available_actions = payload.get("available_actions", None)

                probs = model.action_probs(state=ev.state, spec=spec)
                probs = _mask_and_renorm(probs, available_actions)

                p = float(probs[int(choice)])
                ll += float(np.log(max(p, _EPS)))
                continue

            if ev.type is EventType.OUTCOME:
                payload = ev.payload or {}
                action = payload.get("action", None)
                observed_outcome = payload.get("observed_outcome", None)
                info = payload.get("info", None)

                if action is None:
                    continue

                model.update(
                    state=ev.state,
                    action=int(action),
                    outcome=observed_outcome,
                    spec=spec,
                    info=info,
                )
                continue

            # Other event types are ignored for likelihood by default.

    return ll


def loglike_study_event_log_independent(
    *,
    study: StudyData,
    model: ComputationalModel,
    subject_params: Mapping[str, Mapping[str, float]],
) -> float:
    """Compute total replay log-likelihood for an event-log study.

    Parameters
    ----------
    study : StudyData
        Study dataset containing multiple subjects.
    model : ComputationalModel
        Model instance reused across subjects (parameters are set per subject).
    subject_params : Mapping[str, Mapping[str, float]]
        Mapping from subject_id to that subject's parameter dict.

    Returns
    -------
    float
        Total log-likelihood summed across subjects.
    """
    ll = 0.0
    for subj in study.subjects:
        ll += loglike_subject_event_log(subject=subj, model=model, params=subject_params[subj.subject_id])
    return ll
