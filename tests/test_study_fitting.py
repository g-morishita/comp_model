"""Tests for study-level fitting helpers."""

from __future__ import annotations

import math

import pytest

from comp_model.core.data import BlockData, StudyData, SubjectData, TrialDecision
from comp_model.inference import FitSpec
from comp_model.inference.study_fitting import fit_block_data, fit_study_data, fit_subject_data


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


def test_fit_block_data_returns_block_summary() -> None:
    """Block fitting helper should return block metadata and fit result."""

    block = BlockData(
        block_id="b0",
        trials=(
            _make_trial(0, 1, 1.0),
            _make_trial(1, 0, 0.0),
        ),
    )

    result = fit_block_data(
        block,
        model_component_id="asocial_state_q_value_softmax",
        fit_spec=_fit_spec(),
    )

    assert result.block_id == "b0"
    assert result.n_trials == 2
    assert result.fit_result.best.params == {"alpha": 0.3, "beta": 2.0, "initial_value": 0.0}


def test_fit_subject_data_aggregates_block_results() -> None:
    """Subject fitting helper should aggregate all block fits."""

    subject = SubjectData(
        subject_id="s1",
        blocks=(
            BlockData(block_id="b1", trials=(_make_trial(0, 1, 1.0), _make_trial(1, 1, 1.0))),
            BlockData(block_id="b2", trials=(_make_trial(0, 0, 0.0), _make_trial(1, 1, 1.0))),
        ),
    )

    result = fit_subject_data(
        subject,
        model_component_id="asocial_state_q_value_softmax",
        fit_spec=_fit_spec(),
    )

    assert result.subject_id == "s1"
    assert len(result.block_results) == 2
    assert set(result.mean_best_params) == {"alpha", "beta", "initial_value"}
    assert result.mean_best_params["alpha"] == 0.3
    assert result.mean_best_params["beta"] == 2.0
    assert result.mean_best_params["initial_value"] == 0.0
    assert math.isfinite(result.total_log_likelihood)


def test_fit_study_data_runs_all_subjects() -> None:
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

    result = fit_study_data(
        study,
        model_component_id="asocial_state_q_value_softmax",
        fit_spec=_fit_spec(),
    )

    assert result.n_subjects == 2
    assert len(result.subject_results) == 2
    assert math.isfinite(result.total_log_likelihood)

    subject_ids = {subject.subject_id for subject in result.subject_results}
    assert subject_ids == {"s1", "s2"}


def test_fit_subject_data_supports_joint_block_strategy() -> None:
    """Joint strategy should fit one shared parameter set over all blocks."""

    subject = SubjectData(
        subject_id="s1",
        blocks=(
            BlockData(block_id="b1", trials=(_make_trial(0, 1, 1.0), _make_trial(1, 1, 1.0))),
            BlockData(block_id="b2", trials=(_make_trial(0, 0, 0.0), _make_trial(1, 1, 1.0))),
        ),
    )

    joint = fit_subject_data(
        subject,
        model_component_id="asocial_state_q_value_softmax",
        fit_spec=_fit_spec(),
        block_fit_strategy="joint",
    )
    independent = fit_subject_data(
        subject,
        model_component_id="asocial_state_q_value_softmax",
        fit_spec=_fit_spec(),
        block_fit_strategy="independent",
    )

    assert len(joint.block_results) == 1
    assert joint.block_results[0].block_id == "__joint__"
    assert joint.block_results[0].n_trials == 4
    assert joint.mean_best_params == {"alpha": 0.3, "beta": 2.0, "initial_value": 0.0}
    assert joint.total_log_likelihood == pytest.approx(independent.total_log_likelihood)


def test_fit_subject_data_rejects_unknown_block_fit_strategy() -> None:
    """Subject fitting should reject unsupported block-fit strategy values."""

    subject = SubjectData(
        subject_id="s1",
        blocks=(BlockData(block_id="b1", trials=(_make_trial(0, 1, 1.0),)),),
    )

    with pytest.raises(ValueError, match="block_fit_strategy must be one of"):
        fit_subject_data(
            subject,
            model_component_id="asocial_state_q_value_softmax",
            fit_spec=_fit_spec(),
            block_fit_strategy="bad_mode",  # type: ignore[arg-type]
        )
