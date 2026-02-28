"""Tests for study data containers and trace conversion helpers."""

from __future__ import annotations

import pytest

from comp_model_v2.core.data import (
    BlockData,
    StudyData,
    SubjectData,
    TrialDecision,
    attach_missing_event_traces,
    get_block_trace,
    trace_from_trial_decisions,
    trial_decisions_from_trace,
)
from comp_model_v2.core.events import EventPhase, validate_trace
from comp_model_v2.models import RandomAgent
from comp_model_v2.runtime import replay_episode


def test_trace_from_trial_decisions_builds_canonical_phase_events() -> None:
    """Conversion from decision rows should produce canonical event blocks."""

    rows = (
        TrialDecision(
            trial_index=0,
            decision_index=0,
            actor_id="subject",
            available_actions=(0, 1),
            action=1,
            observation={"trial_index": 0},
            reward=1.0,
        ),
        TrialDecision(
            trial_index=1,
            decision_index=0,
            actor_id="subject",
            available_actions=(0, 1),
            action=0,
            observation={"trial_index": 1},
            reward=0.0,
        ),
    )

    trace = trace_from_trial_decisions(rows)
    validate_trace(trace)

    assert len(trace.events) == 8
    phases = [event.phase for event in trace.by_trial(0)]
    assert phases == [EventPhase.OBSERVATION, EventPhase.DECISION, EventPhase.OUTCOME, EventPhase.UPDATE]


def test_trial_decisions_roundtrip_preserves_multi_decision_rows() -> None:
    """Trace conversion should roundtrip decision metadata and ownership."""

    rows = (
        TrialDecision(
            trial_index=0,
            decision_index=0,
            actor_id="demonstrator",
            learner_id="subject",
            node_id="demo",
            available_actions=(0, 1),
            action=1,
            observation={"stage": "demo"},
            outcome={"reward": 1.0, "source": "demo"},
        ),
        TrialDecision(
            trial_index=0,
            decision_index=1,
            actor_id="subject",
            learner_id="subject",
            node_id="subject",
            available_actions=(0, 1),
            action=0,
            observation={"stage": "subject"},
            outcome={"reward": 0.0, "source": "subject"},
        ),
    )

    trace = trace_from_trial_decisions(rows)
    recovered = trial_decisions_from_trace(trace)

    assert len(recovered) == 2
    assert recovered[0].actor_id == "demonstrator"
    assert recovered[0].learner_id == "subject"
    assert recovered[0].node_id == "demo"
    assert recovered[1].decision_index == 1
    assert recovered[1].node_id == "subject"


def test_attach_missing_event_traces_builds_trace_from_trials() -> None:
    """Study helper should attach traces to blocks missing canonical event logs."""

    block = BlockData(
        block_id="b1",
        trials=(
            TrialDecision(trial_index=0, available_actions=(0, 1), action=0, reward=1.0),
            TrialDecision(trial_index=1, available_actions=(0, 1), action=1, reward=0.0),
        ),
        event_trace=None,
    )
    study = StudyData(subjects=(SubjectData(subject_id="s1", blocks=(block,)),))

    attached = attach_missing_event_traces(study)
    attached_block = attached.subjects[0].blocks[0]

    assert attached_block.event_trace is not None
    validate_trace(attached_block.event_trace)


def test_get_block_trace_prefers_existing_trace_without_regeneration() -> None:
    """Accessor should return existing trace directly when present."""

    block = BlockData(
        block_id="b1",
        trials=(TrialDecision(trial_index=0, available_actions=(0,), action=0, reward=1.0),),
    )
    trace = get_block_trace(block)

    reused = block.with_event_trace(trace)
    assert get_block_trace(reused) is trace


def test_replay_episode_accepts_trace_converted_from_trial_rows() -> None:
    """Converted tabular traces should be replay-compatible."""

    rows = (
        TrialDecision(trial_index=0, available_actions=(0, 1), action=0, reward=1.0),
        TrialDecision(trial_index=1, available_actions=(0, 1), action=1, reward=0.0),
        TrialDecision(trial_index=2, available_actions=(0, 1), action=1, reward=1.0),
    )

    trace = trace_from_trial_decisions(rows)
    replay = replay_episode(trace=trace, model=RandomAgent())

    assert len(replay.steps) == 3
    assert replay.total_log_likelihood == pytest.approx(3 * -0.6931471805599453)


def test_block_data_rejects_invalid_trial_ordering() -> None:
    """Data-layer validation should fail fast on non-canonical row order."""

    with pytest.raises(ValueError, match="sorted"):
        BlockData(
            block_id="b1",
            trials=(
                TrialDecision(trial_index=1, decision_index=0, available_actions=(0,), action=0),
                TrialDecision(trial_index=0, decision_index=0, available_actions=(0,), action=0),
            ),
        )
