"""Tests for study-level fitting helpers."""

from __future__ import annotations

import math

import pytest

from comp_model.core.data import BlockData, StudyData, SubjectData, TrialDecision
from comp_model.inference import FitSpec
from comp_model.inference.fit.group import fit_block, fit_study, fit_subject


def _make_trial(trial_index: int, action: int, reward: float) -> TrialDecision:
    """Build one trial decision row for tests."""

    return TrialDecision(
        trial_index=trial_index,
        decision_index=0,
        actor_id="subject",
        available_actions=(0, 1),
        action=action,
        observation={"state": 0},
        outcome={"reward": reward},
    )


def _fit_spec() -> FitSpec:
    """Return deterministic one-candidate fit spec for asocial state model."""

    return FitSpec(
        solver="grid_search",
        parameter_grid={
            "alpha": [0.3],
            "beta": [2.0],
            "initial_value": [0.0],
        },
    )


def test_fit_block_returns_block_summary() -> None:
    """Block fitting helper should return block metadata and fit result."""

    block = BlockData(
        block_id="b0",
        trials=(
            _make_trial(0, 1, 1.0),
            _make_trial(1, 0, 0.0),
        ),
    )

    result = fit_block(
        block,
        model_component_id="asocial_state_q_value_softmax",
        fit_spec=_fit_spec(),
    )

    assert result.block_id == "b0"
    assert result.n_trials == 2
    assert result.fit_result.best.params == {"alpha": 0.3, "beta": 2.0, "initial_value": 0.0}


def test_fit_subject_independent_returns_one_fit_per_block() -> None:
    """Independent subject fitting should return one estimate per block."""

    subject = SubjectData(
        subject_id="s1",
        blocks=(
            BlockData(block_id="b1", trials=(_make_trial(0, 1, 1.0), _make_trial(1, 1, 1.0))),
            BlockData(block_id="b2", trials=(_make_trial(0, 0, 0.0), _make_trial(1, 1, 1.0))),
        ),
    )

    result = fit_subject(
        subject,
        model_component_id="asocial_state_q_value_softmax",
        fit_spec=_fit_spec(),
    )

    assert result.subject_id == "s1"
    assert len(result.block_results) == 2
    assert result.fit_mode == "independent"
    assert result.shared_best_params is None
    assert result.block_results[0].fit_result.best.params == {
        "alpha": 0.3,
        "beta": 2.0,
        "initial_value": 0.0,
    }
    assert result.block_results[1].fit_result.best.params == {
        "alpha": 0.3,
        "beta": 2.0,
        "initial_value": 0.0,
    }
    assert math.isfinite(result.total_log_likelihood)


def test_fit_study_runs_all_subjects() -> None:
    """Study fitting helper should run one fit per subject and block."""

    study = StudyData(
        subjects=(
            SubjectData(
                subject_id="s1",
                blocks=(
                    BlockData(block_id="b1", trials=(_make_trial(0, 1, 1.0), _make_trial(1, 1, 1.0))),
                ),
            ),
            SubjectData(
                subject_id="s2",
                blocks=(
                    BlockData(block_id="b2", trials=(_make_trial(0, 0, 0.0), _make_trial(1, 0, 0.0))),
                ),
            ),
        )
    )

    result = fit_study(
        study,
        model_component_id="asocial_state_q_value_softmax",
        fit_spec=_fit_spec(),
    )

    assert result.n_subjects == 2
    assert len(result.subject_results) == 2
    assert math.isfinite(result.total_log_likelihood)

    subject_ids = {subject.subject_id for subject in result.subject_results}
    assert subject_ids == {"s1", "s2"}


def test_fit_subject_supports_joint_block_strategy() -> None:
    """Joint strategy should fit one shared parameter set over all blocks."""

    subject = SubjectData(
        subject_id="s1",
        blocks=(
            BlockData(block_id="b1", trials=(_make_trial(0, 1, 1.0), _make_trial(1, 1, 1.0))),
            BlockData(block_id="b2", trials=(_make_trial(0, 0, 0.0), _make_trial(1, 1, 1.0))),
        ),
    )

    joint = fit_subject(
        subject,
        model_component_id="asocial_state_q_value_softmax",
        fit_spec=_fit_spec(),
        block_fit_strategy="joint",
    )
    independent = fit_subject(
        subject,
        model_component_id="asocial_state_q_value_softmax",
        fit_spec=_fit_spec(),
        block_fit_strategy="independent",
    )

    assert len(joint.block_results) == 1
    assert joint.block_results[0].block_id == "__joint__"
    assert joint.block_results[0].n_trials == 4
    assert joint.shared_best_params == {
        "alpha": 0.3,
        "beta": 2.0,
        "initial_value": 0.0,
    }
    assert independent.shared_best_params is None
    assert joint.total_log_likelihood == pytest.approx(independent.total_log_likelihood)


def test_fit_subject_rejects_unknown_block_fit_strategy() -> None:
    """Subject fitting should reject unsupported block-fit strategy values."""

    subject = SubjectData(
        subject_id="s1",
        blocks=(BlockData(block_id="b1", trials=(_make_trial(0, 1, 1.0),)),),
    )

    with pytest.raises(ValueError, match="block_fit_strategy must be one of"):
        fit_subject(
            subject,
            model_component_id="asocial_state_q_value_softmax",
            fit_spec=_fit_spec(),
            block_fit_strategy="bad_mode",  # type: ignore[arg-type]
        )
