"""Tests for Stan data exporters."""

from __future__ import annotations

import pytest

from comp_model_core.data.types import Block, SubjectData, Trial, StudyData
from comp_model_core.events.types import Event, EventLog, EventType
from comp_model_core.spec import EnvironmentSpec, OutcomeType, StateKind

from comp_model_impl.estimators.stan.exporter import (
    _dedupe_preserve_order_str,
    _ensure_int_states,
    _events_for_subject,
    study_to_stan_data,
    study_to_stan_data_within_subject,
    subject_to_stan_data,
    subject_to_stan_data_within_subject,
)


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


def _block(*, block_id: str, condition: str, choice: int) -> Block:
    """Create a single-trial block with a simple event log."""
    log = EventLog(
        events=[
            Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={"condition": condition}),
            Event(idx=1, type=EventType.CHOICE, t=0, state=0, payload={"choice": choice, "available_actions": [0, 1]}),
            Event(
                idx=2,
                type=EventType.OUTCOME,
                t=0,
                state=0,
                payload={"action": choice, "observed_outcome": 1.0, "info": {}},
            ),
        ],
        metadata={"test": True},
    )
    return Block(
        block_id=block_id,
        condition=condition,
        trials=[Trial(t=0, state=0, choice=choice, observed_outcome=1.0, outcome=1.0)],
        env_spec=_spec(),
        event_log=log,
    )


def test_dedupe_preserve_order_str():
    """Deduplication preserves first-seen order."""
    assert _dedupe_preserve_order_str(["A", "B", "A", "C"]) == ["A", "B", "C"]


def test_events_for_subject_concatenates_blocks():
    """Events from blocks are concatenated in order."""
    subj = SubjectData(subject_id="s1", blocks=[_block(block_id="b1", condition="A", choice=0)])
    events = _events_for_subject(subj)
    assert len(events) == 3
    assert events[0].type is EventType.BLOCK_START


def test_ensure_int_states_passes_for_int_states():
    """_ensure_int_states accepts integer-like states."""
    subj = SubjectData(subject_id="s1", blocks=[_block(block_id="b1", condition="A", choice=0)])
    _ensure_int_states(subj)


def test_subject_to_stan_data_basic_fields():
    """subject_to_stan_data produces core Stan arrays."""
    subj = SubjectData(subject_id="s1", blocks=[_block(block_id="b1", condition="A", choice=1)])
    data = subject_to_stan_data(subj)
    assert data["A"] == 2
    assert data["S"] == 1
    assert data["E"] == 3
    assert data["choice"][1] == 2  # 1-indexed
    assert len(data["action_mean"]) == 3
    assert len(data["action_mean"][1]) == 2


def test_subject_to_stan_data_accepts_non_int_states_and_extracts_moments():
    """Exporter should handle dict states and carry action moments."""
    action_moments = [[1.0, 0.25, 0.0], [2.0, 1.0, 0.5]]
    state = {"trial_index": 0, "action_moments": action_moments}
    log = EventLog(
        events=[
            Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={"condition": "A"}),
            Event(
                idx=1,
                type=EventType.CHOICE,
                t=0,
                state=state,
                payload={"choice": 1, "available_actions": [0, 1]},
            ),
            Event(
                idx=2,
                type=EventType.OUTCOME,
                t=0,
                state=state,
                payload={"action": 1, "observed_outcome": 1.0, "info": {"action_moments": action_moments}},
            ),
        ],
        metadata={"test": True},
    )
    block = Block(
        block_id="b1",
        condition="A",
        trials=[Trial(t=0, state=state, choice=1, observed_outcome=1.0, outcome=1.0)],
        env_spec=_spec(),
        event_log=log,
    )
    subj = SubjectData(subject_id="s1", blocks=[block])

    data = subject_to_stan_data(subj)
    assert data["S"] == 2
    assert data["choice"][1] == 2
    assert data["action_mean"][1] == pytest.approx([1.0, 2.0])
    assert data["action_variance"][1] == pytest.approx([0.25, 1.0])
    assert data["action_skewness"][1] == pytest.approx([0.0, 0.5])


def test_study_to_stan_data_aggregates_subjects():
    """study_to_stan_data stacks multiple subjects."""
    s1 = SubjectData(subject_id="s1", blocks=[_block(block_id="b1", condition="A", choice=0)])
    s2 = SubjectData(subject_id="s2", blocks=[_block(block_id="b2", condition="A", choice=1)])
    data = study_to_stan_data(StudyData(subjects=[s1, s2]))
    assert data["N"] == 2
    assert data["E"] == 6
    assert len(data["subj"]) == 6


def test_subject_to_stan_data_within_subject_conditions():
    """Within-subject export includes condition indices and baseline."""
    subj = SubjectData(
        subject_id="s1",
        blocks=[
            _block(block_id="b1", condition="A", choice=0),
            _block(block_id="b2", condition="B", choice=1),
        ],
    )
    data = subject_to_stan_data_within_subject(subj, conditions=["A", "B"], baseline_condition="A")
    assert data["C"] == 2
    assert data["baseline_cond"] == 1
    assert set(data["cond"]) == {1, 2}


def test_study_to_stan_data_within_subject_aggregates():
    """Hier within-subject export stacks subjects and conditions."""
    s1 = SubjectData(subject_id="s1", blocks=[_block(block_id="b1", condition="A", choice=0)])
    s2 = SubjectData(subject_id="s2", blocks=[_block(block_id="b2", condition="A", choice=1)])
    data = study_to_stan_data_within_subject(StudyData(subjects=[s1, s2]), conditions=["A"], baseline_condition="A")
    assert data["N"] == 2
    assert data["C"] == 1
    assert len(data["cond"]) == data["E"]


def test_subject_to_stan_data_within_subject_missing_condition_raises():
    """Missing BLOCK_START condition should raise."""
    log = EventLog(
        events=[
            Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={}),
        ],
        metadata={},
    )
    block = Block(
        block_id="b1",
        condition="A",
        trials=[Trial(t=0, state=0, choice=0, observed_outcome=1.0, outcome=1.0)],
        env_spec=_spec(),
        event_log=log,
    )
    subj = SubjectData(subject_id="s1", blocks=[block])
    with pytest.raises(ValueError):
        subject_to_stan_data_within_subject(subj, conditions=["A"], baseline_condition="A")
