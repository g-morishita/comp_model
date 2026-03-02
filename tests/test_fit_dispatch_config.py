"""Tests for auto-dispatch config fitting helpers."""

from __future__ import annotations

import pytest

import comp_model.inference.config_dispatch as config_dispatch_module
from comp_model.core.data import BlockData, StudyData, SubjectData, TrialDecision
from comp_model.inference import (
    fit_block_auto_from_config,
    fit_dataset_auto_from_config,
    fit_study_auto_from_config,
    fit_subject_auto_from_config,
)


def _trial(trial_index: int, action: int, reward: float) -> TrialDecision:
    """Build one trial row for auto-dispatch tests."""

    return TrialDecision(
        trial_index=trial_index,
        decision_index=0,
        actor_id="subject",
        available_actions=(0, 1),
        action=action,
        observation={"state": 0},
        outcome={"reward": reward},
    )


def _mle_config() -> dict:
    """Build one minimal MLE config."""

    return {
        "model": {"component_id": "asocial_state_q_value_softmax", "kwargs": {}},
        "estimator": {
            "type": "mle", "solver": "grid_search",
            "parameter_grid": {
                "alpha": [0.3],
                "beta": [2.0],
                "initial_value": [0.0],
            },
        },
    }


def _stan_map_config() -> dict:
    """Build one minimal Stan MAP config."""

    return {
        "model": {"component_id": "asocial_state_q_value_softmax", "kwargs": {"initial_value": 0.0}},
        "estimator": {
            "type": "within_subject_hierarchical_stan_map",
            "parameter_names": ["alpha", "beta"],
            "transforms": {
                "alpha": "unit_interval_logit",
                "beta": "positive_log",
            },
            "max_iterations": 50,
            "method": "lbfgs",
            "random_seed": 7,
        },
    }


def test_fit_dataset_auto_dispatches_mle_and_stan_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dataset auto-dispatch should route MLE and Stan MAP fit paths."""

    rows = (_trial(0, 1, 1.0), _trial(1, 0, 0.0), _trial(2, 1, 1.0))

    mle_result = fit_dataset_auto_from_config(rows, config=_mle_config())
    assert hasattr(mle_result, "best")

    sentinel = object()
    monkeypatch.setattr(
        config_dispatch_module,
        "sample_subject_hierarchical_posterior_from_config",
        lambda *args, **kwargs: sentinel,
    )
    map_result = fit_dataset_auto_from_config(rows, config=_stan_map_config())
    assert map_result is sentinel


def test_fit_block_subject_study_auto_dispatch_stan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Block/subject/study auto-dispatch should route Stan estimators."""

    block = BlockData(
        block_id="b0",
        trials=(_trial(0, 1, 1.0), _trial(1, 0, 0.0), _trial(2, 1, 1.0)),
    )
    subject = SubjectData(subject_id="s1", blocks=(block,))
    study = StudyData(subjects=(subject,))

    sentinel_subject = object()
    sentinel_study = object()
    monkeypatch.setattr(
        config_dispatch_module,
        "sample_subject_hierarchical_posterior_from_config",
        lambda *args, **kwargs: sentinel_subject,
    )
    monkeypatch.setattr(
        config_dispatch_module,
        "sample_study_hierarchical_posterior_from_config",
        lambda *args, **kwargs: sentinel_study,
    )

    block_result = fit_block_auto_from_config(block, config=_stan_map_config())
    assert block_result is sentinel_subject

    subject_result = fit_subject_auto_from_config(subject, config=_stan_map_config())
    assert subject_result is sentinel_subject

    study_result = fit_study_auto_from_config(study, config=_stan_map_config())
    assert study_result is sentinel_study


def test_fit_auto_rejects_legacy_scipy_map_estimator() -> None:
    """Auto-dispatch should reject removed SciPy Bayesian estimator types."""

    rows = (_trial(0, 1, 1.0),)
    config = {
        "model": {"component_id": "asocial_state_q_value_softmax", "kwargs": {}},
        "estimator": {"type": "scipy_map"},
    }

    with pytest.raises(ValueError, match="unsupported estimator.type"):
        fit_dataset_auto_from_config(rows, config=config)

