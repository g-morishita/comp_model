"""Tests for legacy Bayesian config API behavior."""

from __future__ import annotations

import pytest

from comp_model.core.data import BlockData, StudyData, SubjectData, TrialDecision
from comp_model.inference.bayes_config import (
    fit_map_block_from_config,
    fit_map_dataset_from_config,
    fit_map_study_from_config,
    fit_map_subject_from_config,
    fit_study_hierarchical_map_from_config,
    fit_subject_hierarchical_map_from_config,
    hierarchical_map_spec_from_config,
    map_fit_spec_from_config,
    prior_program_from_config,
)


def _trial(trial_index: int, action: int, reward: float) -> TrialDecision:
    """Build one trial row for config tests."""

    return TrialDecision(
        trial_index=trial_index,
        decision_index=0,
        actor_id="subject",
        available_actions=(0, 1),
        action=action,
        observation={"state": 0},
        outcome={"reward": reward},
    )


def _scipy_map_config() -> dict:
    """Build one removed SciPy MAP config."""

    return {
        "model": {"component_id": "asocial_state_q_value_softmax", "kwargs": {}},
        "prior": {
            "type": "independent",
            "parameters": {
                "alpha": {"distribution": "uniform", "lower": 0.0, "upper": 1.0},
            },
        },
        "estimator": {
            "type": "scipy_map",
            "initial_params": {"alpha": 0.5},
            "bounds": {"alpha": [0.0, 1.0]},
        },
    }


def test_map_fit_spec_from_config_rejects_removed_scipy_estimators() -> None:
    """Legacy SciPy MAP parser should reject removed estimator types."""

    with pytest.raises(ValueError, match="no longer supported"):
        map_fit_spec_from_config(_scipy_map_config()["estimator"])


def test_hierarchical_map_spec_from_config_rejects_removed_estimator_type() -> None:
    """Legacy hierarchical MAP parser should reject removed estimator type."""

    with pytest.raises(ValueError, match="has been removed"):
        hierarchical_map_spec_from_config(
            {
                "type": "within_subject_hierarchical_map",
                "parameter_names": ["alpha"],
            }
        )


def test_prior_program_from_config_still_supports_independent_priors() -> None:
    """Independent prior parsing should remain available for utility use."""

    prior = prior_program_from_config(
        {
            "type": "independent",
            "parameters": {
                "alpha": {"distribution": "uniform", "lower": 0.0, "upper": 1.0},
            },
        }
    )
    assert prior.log_prior({"alpha": 0.3}) == pytest.approx(0.0)


def test_removed_map_fit_helpers_raise_runtime() -> None:
    """Legacy fit_map_* config entry points should fail with migration guidance."""

    rows = (_trial(0, 1, 1.0),)
    block = BlockData(block_id="b0", trials=rows)
    subject = SubjectData(subject_id="s1", blocks=(block,))
    study = StudyData(subjects=(subject,))
    config = _scipy_map_config()

    with pytest.raises(RuntimeError, match="has been removed"):
        fit_map_dataset_from_config(rows, config=config)
    with pytest.raises(RuntimeError, match="has been removed"):
        fit_map_block_from_config(block, config=config)
    with pytest.raises(RuntimeError, match="has been removed"):
        fit_map_subject_from_config(subject, config=config)
    with pytest.raises(RuntimeError, match="has been removed"):
        fit_map_study_from_config(study, config=config)


def test_removed_legacy_hierarchical_map_helpers_raise_runtime() -> None:
    """Legacy hierarchical MAP config entry points should fail fast."""

    block = BlockData(block_id="b0", trials=(_trial(0, 1, 1.0),))
    subject = SubjectData(subject_id="s1", blocks=(block,))
    study = StudyData(subjects=(subject,))
    config = {
        "model": {"component_id": "asocial_state_q_value_softmax", "kwargs": {}},
        "estimator": {
            "type": "within_subject_hierarchical_map",
            "parameter_names": ["alpha"],
        },
    }

    with pytest.raises(RuntimeError, match="has been removed"):
        fit_subject_hierarchical_map_from_config(subject, config=config)
    with pytest.raises(RuntimeError, match="has been removed"):
        fit_study_hierarchical_map_from_config(study, config=config)

