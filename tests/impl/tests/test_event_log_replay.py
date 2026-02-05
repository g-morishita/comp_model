"""Tests for event-log replay likelihood helpers."""

from __future__ import annotations

import numpy as np
import pytest

from comp_model_core.data.types import Block, SubjectData, Trial, StudyData
from comp_model_core.events.types import Event, EventLog, EventType
from comp_model_core.spec import EnvironmentSpec, OutcomeType, StateKind

from comp_model_impl.likelihood.event_log_replay import (
    _mask_and_renorm,
    loglike_study_independent,
    loglike_subject,
)
from comp_model_impl.models.qrl.qrl import QRL


def _spec() -> EnvironmentSpec:
    """Return a minimal binary environment spec."""
    return EnvironmentSpec(
        n_actions=2,
        outcome_type=OutcomeType.BINARY,
        outcome_range=(0.0, 1.0),
        outcome_is_bounded=True,
        is_social=False,
        state_kind=StateKind.DISCRETE,
        n_states=1,
    )


def _block_with_events(*, choice: int, condition: str = "c1", include_social: bool = False) -> Block:
    """Create a single-trial block with a simple event log."""
    events = [
        Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={"condition": condition}),
    ]
    if include_social:
        events.append(
            Event(
                idx=1,
                type=EventType.SOCIAL_OBSERVED,
                t=0,
                state=0,
                payload={"others_choices": [0], "observed_others_outcomes": [1.0], "social_info": {}},
            )
        )
        idx = 2
    else:
        idx = 1

    events.append(
        Event(
            idx=idx,
            type=EventType.CHOICE,
            t=0,
            state=0,
            payload={"choice": choice, "available_actions": [0, 1]},
        )
    )
    events.append(
        Event(
            idx=idx + 1,
            type=EventType.OUTCOME,
            t=0,
            state=0,
            payload={"action": choice, "observed_outcome": 1.0, "info": {}},
        )
    )
    log = EventLog(events=events, metadata={"test": True})
    return Block(
        block_id="b1",
        condition=condition,
        trials=[Trial(t=0, state=0, choice=choice, observed_outcome=1.0, outcome=1.0)],
        env_spec=_spec(),
        event_log=log,
    )


def test_mask_and_renorm_masks_and_scales():
    """_mask_and_renorm removes disallowed actions and renormalizes."""
    probs = np.array([0.2, 0.3, 0.5], dtype=float)
    out = _mask_and_renorm(probs, available_actions=[0, 2])
    assert out[1] == 0.0
    assert out.sum() == pytest.approx(1.0)


def test_loglike_subject_missing_condition_raises():
    """BLOCK_START without condition should raise a ValueError."""
    log = EventLog(events=[Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={})], metadata={})
    block = Block(
        block_id="b1",
        condition="c1",
        trials=[Trial(t=0, state=0, choice=0, observed_outcome=1.0, outcome=1.0)],
        env_spec=_spec(),
        event_log=log,
    )
    subj = SubjectData(subject_id="s1", blocks=[block])
    with pytest.raises(ValueError):
        loglike_subject(subject=subj, model=QRL(), params={"alpha": 0.2, "beta": 2.0})


def test_loglike_subject_ignores_social_events_for_asocial_model():
    """Asocial models should ignore SOCIAL_OBSERVED events."""
    block = _block_with_events(choice=1, include_social=True)
    subj = SubjectData(subject_id="s1", blocks=[block])
    ll = loglike_subject(subject=subj, model=QRL(), params={"alpha": 0.2, "beta": 1.0})
    assert ll == pytest.approx(np.log(0.5), abs=1e-12)


def test_loglike_study_independent_sums_subjects():
    """Study likelihood is the sum of per-subject likelihoods."""
    s1 = SubjectData(subject_id="s1", blocks=[_block_with_events(choice=0)])
    s2 = SubjectData(subject_id="s2", blocks=[_block_with_events(choice=1)])
    study = StudyData(subjects=[s1, s2])

    model = QRL()
    params = {"alpha": 0.2, "beta": 1.0}
    ll1 = loglike_subject(subject=s1, model=model, params=params)
    ll2 = loglike_subject(subject=s2, model=model, params=params)
    ll = loglike_study_independent(study=study, model=model, subject_params={"s1": params, "s2": params})
    assert ll == pytest.approx(ll1 + ll2, abs=1e-12)
