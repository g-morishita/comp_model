"""comp_model_impl.likelihood.replay

Replay likelihood for trial-by-trial datasets.

This module computes log-likelihood by replaying recorded choices through a model:

- The model generates action probabilities for each trial state.
- Probabilities are masked/renormalized using any recorded ``available_actions``.
- The probability assigned to the observed choice contributes to log-likelihood.
- The model updates using the recorded observed outcome (which may be None if hidden).

Social data support
-------------------
If the dataset contains demonstrator signals (choices/outcomes) and the model is a
:class:`comp_model_core.interfaces.model.SocialComputationalModel`, the replay includes
calls to ``model.social_update(...)`` on each trial where social data are present.

See Also
--------
comp_model_core.data.types.SubjectData
comp_model_core.data.types.StudyData
comp_model_core.interfaces.model.ComputationalModel
comp_model_core.interfaces.model.SocialComputationalModel
"""

from __future__ import annotations

import numpy as np
from typing import Mapping, Sequence

from comp_model_core.data.types import StudyData, SubjectData
from comp_model_core.interfaces.model import ComputationalModel, SocialComputationalModel
from comp_model_core.interfaces.block_runner import SocialObservation

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
        Masked and renormalized probabilities. Sums to 1 (up to numerical tolerance).
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


def loglike_subject(
    *,
    subject: SubjectData,
    model: ComputationalModel,
    params: Mapping[str, float],
) -> float:
    """Compute replay log-likelihood for a single subject.

    Parameters
    ----------
    subject : SubjectData
        Subject dataset containing blocks and trials.
    model : ComputationalModel
        Model instance used to compute probabilities and updates.
    params : Mapping[str, float]
        Model parameters for this subject.

    Returns
    -------
    float
        Total log-likelihood across all choices in the subject dataset.

    Raises
    ------
    ValueError
        If a block is missing ``env_spec`` (required for replay).
    """
    model.set_params(params)
    ll = 0.0

    is_social_model = isinstance(model, SocialComputationalModel)

    for block in subject.blocks:
        spec = block.env_spec
        if spec is None:
            raise ValueError("Block.env_spec is None; required for replay.")

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
            probs = _mask_and_renorm(probs, tr.available_actions)

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
    """Compute total replay log-likelihood for a study (independent subjects).

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
        Sum of subject log-likelihoods.
    """
    ll = 0.0
    for subj in study.subjects:
        ll += loglike_subject(subject=subj, model=model, params=subject_params[subj.subject_id])
    return ll
