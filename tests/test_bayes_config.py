"""Tests for config-driven Bayesian fitting helpers."""

from __future__ import annotations

import math

import pytest

from comp_model.core.data import BlockData, StudyData, SubjectData, TrialDecision
from comp_model.inference import (
    fit_map_dataset_from_config,
    fit_study_hierarchical_map_from_config,
    fit_subject_hierarchical_map_from_config,
    map_fit_spec_from_config,
    prior_program_from_config,
)


def _trial(trial_index: int, action: int, reward: float) -> TrialDecision:
    """Build one trial row for Bayesian config tests."""

    return TrialDecision(
        trial_index=trial_index,
        decision_index=0,
        actor_id="subject",
        available_actions=(0, 1),
        action=action,
        observation={"state": 0},
        outcome={"reward": reward},
    )


def test_map_fit_spec_from_config_transformed() -> None:
    """MAP estimator parser should construct transformed MAP fit specs."""

    spec = map_fit_spec_from_config(
        {
            "type": "transformed_scipy_map",
            "initial_params": {"alpha": 0.2},
            "bounds_z": {"alpha": [-5.0, 5.0]},
            "transforms": {"alpha": "unit_interval_logit"},
            "method": "L-BFGS-B",
            "tol": 1e-6,
        }
    )

    assert spec.estimator_type == "transformed_scipy_map"
    assert spec.initial_params == {"alpha": 0.2}
    assert spec.bounds_z == {"alpha": (-5.0, 5.0)}
    assert spec.transforms is not None
    assert "alpha" in spec.transforms
    assert spec.method == "L-BFGS-B"
    assert spec.tol == pytest.approx(1e-6)


def test_prior_program_from_config_rejects_unknown_distribution() -> None:
    """Prior parser should fail on unsupported distribution names."""

    with pytest.raises(ValueError, match="unsupported prior distribution"):
        prior_program_from_config(
            {
                "type": "independent",
                "parameters": {
                    "alpha": {"distribution": "not_a_distribution"},
                },
            }
        )


def test_fit_map_dataset_from_config_runs_end_to_end() -> None:
    """Config-driven MAP fit should run on trial-row datasets."""

    rows = (_trial(0, 1, 1.0), _trial(1, 0, 0.0), _trial(2, 1, 1.0), _trial(3, 1, 1.0))
    config = {
        "model": {
            "component_id": "asocial_state_q_value_softmax",
            "kwargs": {},
        },
        "prior": {
            "type": "independent",
            "parameters": {
                "alpha": {"distribution": "uniform", "lower": 0.0, "upper": 1.0},
                "beta": {"distribution": "uniform", "lower": 0.0, "upper": 20.0},
                "initial_value": {"distribution": "normal", "mean": 0.0, "std": 1.0},
            },
        },
        "estimator": {
            "type": "scipy_map",
            "initial_params": {"alpha": 0.5, "beta": 1.0, "initial_value": 0.0},
            "bounds": {
                "alpha": [0.0, 1.0],
                "beta": [0.0, 20.0],
                "initial_value": [None, None],
            },
        },
    }

    result = fit_map_dataset_from_config(rows, config=config)
    assert set(result.map_params) == {"alpha", "beta", "initial_value"}
    assert math.isfinite(result.map_candidate.log_likelihood)
    assert math.isfinite(result.map_candidate.log_posterior)


def test_hierarchical_map_from_config_runs_subject_and_study() -> None:
    """Hierarchical config runners should support subject and study inputs."""

    subject_1 = SubjectData(
        subject_id="s1",
        blocks=(
            BlockData(block_id="b1", trials=(_trial(0, 1, 1.0), _trial(1, 0, 0.0), _trial(2, 1, 1.0))),
            BlockData(block_id="b2", trials=(_trial(0, 0, 0.0), _trial(1, 1, 1.0), _trial(2, 1, 1.0))),
        ),
    )
    subject_2 = SubjectData(
        subject_id="s2",
        blocks=(
            BlockData(block_id="b1", trials=(_trial(0, 0, 0.0), _trial(1, 0, 0.0), _trial(2, 1, 1.0))),
            BlockData(block_id="b2", trials=(_trial(0, 1, 1.0), _trial(1, 1, 1.0), _trial(2, 1, 1.0))),
        ),
    )

    config = {
        "model": {
            "component_id": "asocial_state_q_value_softmax",
            "kwargs": {},
        },
        "estimator": {
            "type": "within_subject_hierarchical_map",
            "parameter_names": ["alpha", "beta", "initial_value"],
            "transforms": {
                "alpha": "unit_interval_logit",
                "beta": "positive_log",
                "initial_value": "identity",
            },
            "initial_group_location": {
                "alpha": 0.4,
                "beta": 1.0,
                "initial_value": 0.0,
            },
            "initial_group_scale": {
                "alpha": 0.5,
                "beta": 0.5,
                "initial_value": 0.5,
            },
            "initial_block_params_by_subject": {
                "s1": [
                    {"alpha": 0.4, "beta": 1.0, "initial_value": 0.0},
                    {"alpha": 0.4, "beta": 1.0, "initial_value": 0.0},
                ]
            },
            "mu_prior_mean": 0.0,
            "mu_prior_std": 2.0,
            "log_sigma_prior_mean": -1.0,
            "log_sigma_prior_std": 1.0,
            "method": "L-BFGS-B",
        },
    }

    subject_result = fit_subject_hierarchical_map_from_config(subject_1, config=config)
    assert subject_result.subject_id == "s1"
    assert len(subject_result.block_results) == 2
    assert math.isfinite(subject_result.total_log_posterior)

    study_result = fit_study_hierarchical_map_from_config(
        StudyData(subjects=(subject_1, subject_2)),
        config=config,
    )
    assert study_result.n_subjects == 2
    assert len(study_result.subject_results) == 2
    assert math.isfinite(study_result.total_log_posterior)
