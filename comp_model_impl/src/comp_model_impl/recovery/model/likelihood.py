"""Likelihood utilities for model recovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from comp_model_core.data.types import StudyData, SubjectData
from comp_model_core.events.accessors import get_event_log
from comp_model_core.events.types import EventType
from comp_model_core.interfaces.model import ComputationalModel

from comp_model_impl.likelihood.event_log_replay import loglike_subject


@dataclass(frozen=True, slots=True)
class LikelihoodSummary:
    """Likelihood and observation counts for one fitted model.

    Attributes
    ----------
    ll_total : float
        Total log-likelihood summed over subjects.
    ll_by_subject : dict[str, float]
        Per-subject log-likelihood.
    n_obs_total : int
        Total number of choice observations.
    n_obs_by_subject : dict[str, int]
        Per-subject choice-observation counts.
    """
    ll_total: float
    ll_by_subject: dict[str, float]
    n_obs_total: int
    n_obs_by_subject: dict[str, int]


def _count_choice_events(subject: SubjectData) -> int:
    """Count choice events for one subject.

    Parameters
    ----------
    subject : SubjectData
        Subject data containing block event logs.

    Returns
    -------
    int
        Number of ``EventType.CHOICE`` events.
    """

    n = 0
    for blk in subject.blocks:
        evlog = get_event_log(blk)
        for e in evlog.events:
            if e.type is EventType.CHOICE:
                n += 1
    return int(n)


def compute_likelihood_summary(
    *,
    study: StudyData,
    model: ComputationalModel,
    subject_params: Mapping[str, Mapping[str, float]],
) -> LikelihoodSummary:
    """Compute total/per-subject log-likelihood and choice-event counts.

    Parameters
    ----------
    study : StudyData
        Study data with event logs on blocks.
    model : ComputationalModel
        Candidate model (will be deep-copied during replay).
    subject_params : Mapping[str, Mapping[str, float]]
        Mapping subject_id -> {param: value} for the candidate model.

    Returns
    -------
    LikelihoodSummary
        Aggregated log-likelihood and observation counts.
    """
    ll_by: dict[str, float] = {}
    n_by: dict[str, int] = {}
    ll_total = 0.0
    n_total = 0

    # StudyData.subjects is a list; align by subject_id string.
    for subj in study.subjects:
        sid = str(subj.subject_id)
        n_obs = _count_choice_events(subj)
        n_by[sid] = n_obs
        n_total += n_obs

        params = subject_params.get(sid, None)
        if params is None:
            # Missing fits: treat as -inf likelihood (but keep counts).
            ll = float("-inf")
        else:
            try:
                ll = float(loglike_subject(subject=subj, model=model, params=params))
            except Exception:
                ll = float("-inf")
        ll_by[sid] = ll
        ll_total += ll

    return LikelihoodSummary(
        ll_total=float(ll_total),
        ll_by_subject=ll_by,
        n_obs_total=int(n_total),
        n_obs_by_subject=n_by,
    )
