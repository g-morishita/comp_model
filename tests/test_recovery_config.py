"""Tests for config-driven recovery runners."""

from __future__ import annotations

import json

import pytest

from comp_model.recovery import (
    load_json_config,
    run_model_recovery_from_config,
    run_parameter_recovery_from_config,
)


def test_parameter_recovery_runs_from_json_config(tmp_path) -> None:
    """Parameter recovery should run from a serialized JSON configuration."""

    config = {
        "problem": {
            "component_id": "stationary_bandit",
            "kwargs": {"reward_probabilities": [0.2, 0.8]},
        },
        "generating_model": {
            "component_id": "asocial_state_q_value_softmax",
            "kwargs": {},
        },
        "fitting_model": {
            "component_id": "asocial_state_q_value_softmax",
            "kwargs": {},
        },
        "estimator": {
            "type": "grid_search",
            "parameter_grid": {
                "alpha": [0.3],
                "beta": [2.0],
                "initial_value": [0.0],
            },
        },
        "true_parameter_sets": [
            {"alpha": 0.3, "beta": 2.0, "initial_value": 0.0},
        ],
        "n_trials": 30,
        "seed": 11,
    }

    path = tmp_path / "param_recovery_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    loaded = load_json_config(path)
    result = run_parameter_recovery_from_config(loaded)

    assert len(result.cases) == 1
    case = result.cases[0]
    assert case.true_params == {"alpha": 0.3, "beta": 2.0, "initial_value": 0.0}
    assert case.estimated_params == {"alpha": 0.3, "beta": 2.0, "initial_value": 0.0}
    assert result.mean_absolute_error == {"alpha": 0.0, "beta": 0.0, "initial_value": 0.0}


def test_model_recovery_runs_from_mapping_config() -> None:
    """Model recovery should run from mapping config and emit confusion counts."""

    config = {
        "problem": {
            "component_id": "stationary_bandit",
            "kwargs": {"reward_probabilities": [0.2, 0.8]},
        },
        "generating": [
            {
                "name": "qrl_generator",
                "model": {
                    "component_id": "asocial_state_q_value_softmax",
                    "kwargs": {},
                },
                "true_params": {"alpha": 0.3, "beta": 2.0, "initial_value": 0.0},
            },
        ],
        "candidates": [
            {
                "name": "candidate_good",
                "model": {
                    "component_id": "asocial_state_q_value_softmax",
                    "kwargs": {},
                },
                "estimator": {
                    "type": "grid_search",
                    "parameter_grid": {
                        "alpha": [0.3],
                        "beta": [2.0],
                        "initial_value": [0.0],
                    },
                },
                "n_parameters": 3,
            },
            {
                "name": "candidate_bad",
                "model": {
                    "component_id": "asocial_state_q_value_softmax",
                    "kwargs": {},
                },
                "estimator": {
                    "type": "grid_search",
                    "parameter_grid": {
                        "alpha": [0.9],
                        "beta": [0.1],
                        "initial_value": [1.0],
                    },
                },
                "n_parameters": 3,
            },
        ],
        "n_trials": 40,
        "n_replications_per_generator": 3,
        "criterion": "log_likelihood",
        "seed": 9,
    }

    result = run_model_recovery_from_config(config)

    assert len(result.cases) == 3
    assert result.criterion == "log_likelihood"
    assert set(result.confusion_matrix["qrl_generator"]).issubset({"candidate_good", "candidate_bad"})
    assert sum(result.confusion_matrix["qrl_generator"].values()) == 3

    for case in result.cases:
        assert case.generating_model_name == "qrl_generator"
        assert len(case.candidate_summaries) == 2


def test_recovery_config_rejects_unknown_estimator_type() -> None:
    """Config runner should fail on unsupported estimator type."""

    config = {
        "problem": {
            "component_id": "stationary_bandit",
            "kwargs": {"reward_probabilities": [0.5, 0.5]},
        },
        "generating_model": {
            "component_id": "asocial_state_q_value_softmax",
            "kwargs": {},
        },
        "fitting_model": {
            "component_id": "asocial_state_q_value_softmax",
            "kwargs": {},
        },
        "estimator": {
            "type": "unknown_estimator",
        },
        "true_parameter_sets": [
            {"alpha": 0.2, "beta": 1.0, "initial_value": 0.0},
        ],
        "n_trials": 10,
    }

    with pytest.raises(ValueError, match="must be one of"):
        run_parameter_recovery_from_config(config)
