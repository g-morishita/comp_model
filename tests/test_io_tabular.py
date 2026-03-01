"""Tests for tabular CSV I/O helpers."""

from __future__ import annotations

from comp_model.core.data import BlockData, StudyData, SubjectData, TrialDecision, trace_from_trial_decisions
from comp_model.io import (
    read_study_decisions_csv,
    read_trial_decisions_csv,
    write_study_decisions_csv,
    write_trial_decisions_csv,
)


def _trial(
    trial_index: int,
    decision_index: int,
    action: int,
    reward: float,
    *,
    actor_id: str = "subject",
) -> TrialDecision:
    """Build one trial row for I/O tests."""

    return TrialDecision(
        trial_index=trial_index,
        decision_index=decision_index,
        actor_id=actor_id,
        learner_id="subject",
        node_id=f"node_{decision_index}",
        available_actions=(0, 1),
        action=action,
        observation={"state": 0, "trial": trial_index},
        outcome={"reward": reward},
        reward=reward,
        metadata={"source": "test"},
    )


def test_trial_decision_csv_roundtrip(tmp_path) -> None:
    """Trial decision CSV writer/reader should round-trip core fields."""

    rows = (
        _trial(0, 0, 1, 1.0),
        _trial(1, 0, 0, 0.0),
        _trial(2, 0, 1, 1.0),
    )
    path = tmp_path / "trial_decisions.csv"
    out = write_trial_decisions_csv(rows, path)

    loaded = read_trial_decisions_csv(out)
    assert len(loaded) == 3
    assert loaded[0].trial_index == 0
    assert loaded[0].available_actions == (0, 1)
    assert loaded[0].action == 1
    assert loaded[0].observation == {"state": 0, "trial": 0}
    assert loaded[0].outcome == {"reward": 1.0}
    assert loaded[0].metadata == {"source": "test"}


def test_study_decision_csv_roundtrip(tmp_path) -> None:
    """Study CSV writer/reader should preserve subject/block grouping."""

    study = StudyData(
        subjects=(
            SubjectData(
                subject_id="s1",
                blocks=(
                    BlockData(block_id=1, trials=(_trial(0, 0, 1, 1.0), _trial(1, 0, 0, 0.0))),
                    BlockData(block_id="b2", trials=(_trial(0, 0, 1, 1.0),)),
                ),
            ),
            SubjectData(
                subject_id="s2",
                blocks=(
                    BlockData(block_id="b1", trials=(_trial(0, 0, 0, 0.0),)),
                ),
            ),
        )
    )
    path = tmp_path / "study.csv"
    out = write_study_decisions_csv(study, path)
    loaded = read_study_decisions_csv(out)

    assert loaded.n_subjects == 2
    assert loaded.subjects[0].subject_id == "s1"
    assert len(loaded.subjects[0].blocks) == 2
    assert loaded.subjects[0].blocks[0].block_id == 1
    assert loaded.subjects[0].blocks[1].block_id == "b2"
    assert loaded.subjects[1].subject_id == "s2"
    assert loaded.subjects[0].blocks[0].trials[0].action == 1


def test_write_study_decisions_csv_uses_event_trace_when_trials_missing(tmp_path) -> None:
    """Study CSV writer should flatten block event traces when trials are missing."""

    trial_rows = (
        _trial(0, 0, 1, 1.0),
        _trial(1, 0, 0, 0.0),
    )
    trace_block = BlockData(block_id="b0", event_trace=trace_from_trial_decisions(trial_rows))
    study = StudyData(subjects=(SubjectData(subject_id="s1", blocks=(trace_block,)),))

    path = tmp_path / "trace_study.csv"
    out = write_study_decisions_csv(study, path)
    loaded = read_study_decisions_csv(out)

    assert loaded.n_subjects == 1
    assert len(loaded.subjects[0].blocks[0].trials) == 2
    assert loaded.subjects[0].blocks[0].trials[0].action == 1


def test_read_trial_decisions_csv_rejects_missing_required_columns(tmp_path) -> None:
    """Trial reader should fail fast when required columns are missing."""

    path = tmp_path / "invalid.csv"
    path.write_text("trial_index\n0\n", encoding="utf-8")

    try:
        read_trial_decisions_csv(path)
    except ValueError as exc:
        assert "missing required columns" in str(exc)
    else:  # pragma: no cover - explicit failure path
        raise AssertionError("expected ValueError for missing required columns")
