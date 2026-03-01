"""Tests for config-driven MCMC posterior sampling helpers."""

from __future__ import annotations

import pytest

from comp_model.core.data import BlockData, TrialDecision
from comp_model.inference import (
    fit_block_auto_from_config,
    fit_dataset_auto_from_config,
    mcmc_estimator_spec_from_config,
    sample_posterior_dataset_from_config,
)


def _trial(trial_index: int, action: int, reward: float) -> TrialDecision:
    """Build one trial row for MCMC config tests."""

    return TrialDecision(
        trial_index=trial_index,
        decision_index=0,
        actor_id="subject",
        available_actions=(0, 1),
        action=action,
        observation={"state": 0},
        outcome={"reward": reward},
    )


def _mcmc_config() -> dict:
    """Build one minimal MCMC config."""

    return {
        "model": {"component_id": "asocial_state_q_value_softmax", "kwargs": {}},
        "prior": {
            "type": "independent",
            "parameters": {
                "alpha": {"distribution": "uniform", "lower": 0.0, "upper": 1.0},
                "beta": {"distribution": "uniform", "lower": 0.0, "upper": 20.0},
                "initial_value": {"distribution": "normal", "mean": 0.0, "std": 1.0},
            },
        },
        "estimator": {
            "type": "random_walk_metropolis",
            "initial_params": {"alpha": 0.4, "beta": 2.0, "initial_value": 0.0},
            "n_samples": 20,
            "n_warmup": 20,
            "thin": 1,
            "proposal_scales": {"alpha": 0.05, "beta": 0.2, "initial_value": 0.1},
            "bounds": {
                "alpha": [0.0, 1.0],
                "beta": [0.0, 20.0],
                "initial_value": [None, None],
            },
            "random_seed": 9,
        },
    }


def test_mcmc_estimator_spec_from_config_parses_fields() -> None:
    """MCMC estimator parser should construct a full spec."""

    spec = mcmc_estimator_spec_from_config(_mcmc_config()["estimator"])
    assert spec.n_samples == 20
    assert spec.n_warmup == 20
    assert spec.thin == 1
    assert spec.initial_params["alpha"] == pytest.approx(0.4)
    assert spec.proposal_scales is not None
    assert spec.proposal_scales["beta"] == pytest.approx(0.2)
    assert spec.bounds is not None
    assert spec.bounds["alpha"] == (0.0, 1.0)
    assert spec.random_seed == 9


def test_sample_posterior_dataset_from_config_runs_end_to_end() -> None:
    """Config-driven MCMC helper should return posterior samples."""

    rows = (_trial(0, 1, 1.0), _trial(1, 0, 0.0), _trial(2, 1, 1.0), _trial(3, 1, 1.0))
    result = sample_posterior_dataset_from_config(rows, config=_mcmc_config())

    assert result.posterior_samples.n_draws == 20
    assert set(result.posterior_samples.parameter_names) == {
        "alpha",
        "beta",
        "initial_value",
    }


def test_fit_dataset_auto_dispatches_mcmc() -> None:
    """Dataset auto-dispatch should route to MCMC posterior sampling."""

    rows = (_trial(0, 1, 1.0), _trial(1, 0, 0.0), _trial(2, 1, 1.0))
    result = fit_dataset_auto_from_config(rows, config=_mcmc_config())
    assert result.posterior_samples.n_draws == 20


def test_fit_block_auto_rejects_mcmc_estimator_type() -> None:
    """Block auto-dispatch should reject dataset-only MCMC estimator type."""

    block = BlockData(
        block_id="b0",
        trials=(_trial(0, 1, 1.0), _trial(1, 0, 0.0)),
    )
    with pytest.raises(ValueError, match="dataset-level fitting only"):
        fit_block_auto_from_config(block, config=_mcmc_config())
