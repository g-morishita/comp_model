"""Tests for config-driven recovery runners."""

from __future__ import annotations

import json

import numpy as np
import pytest

from comp_model.recovery import (
    load_config,
    load_json_config,
    run_model_recovery_from_config,
    run_parameter_recovery_from_config,
)
from comp_model.recovery.config import _sample_true_parameter_sets_from_sampling


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
            "type": "mle", "solver": "grid_search",
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


def test_parameter_recovery_runs_from_yaml_config(tmp_path) -> None:
    """Parameter recovery should run from YAML configuration files."""

    path = tmp_path / "param_recovery_config.yaml"
    path.write_text(
        "\n".join(
            [
                "problem:",
                "  component_id: stationary_bandit",
                "  kwargs:",
                "    reward_probabilities: [0.2, 0.8]",
                "generating_model:",
                "  component_id: asocial_state_q_value_softmax",
                "  kwargs: {}",
                "fitting_model:",
                "  component_id: asocial_state_q_value_softmax",
                "  kwargs: {}",
                "estimator:",
                "  type: mle",
                "  solver: grid_search",
                "  parameter_grid:",
                "    alpha: [0.3]",
                "    beta: [2.0]",
                "    initial_value: [0.0]",
                "true_parameter_sets:",
                "  - alpha: 0.3",
                "    beta: 2.0",
                "    initial_value: 0.0",
                "n_trials: 30",
                "seed: 11",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_config(path)
    result = run_parameter_recovery_from_config(loaded)

    assert len(result.cases) == 1
    assert result.cases[0].estimated_params == {"alpha": 0.3, "beta": 2.0, "initial_value": 0.0}


def test_parameter_recovery_samples_true_params_from_distributions() -> None:
    """Parameter recovery should support sampling true parameters from config distributions."""

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
            "type": "mle", "solver": "grid_search",
            "parameter_grid": {
                "alpha": [0.2, 0.4, 0.6, 0.8],
                "beta": [1.0, 2.0, 3.0],
                "initial_value": [0.0],
            },
        },
        "true_parameter_distributions": {
            "alpha": {"distribution": "uniform", "lower": 0.0, "upper": 1.0},
            "beta": {"distribution": "uniform", "lower": 0.1, "upper": 5.0},
            "initial_value": {"distribution": "normal", "mean": 0.0, "std": 0.2},
        },
        "n_parameter_sets": 4,
        "n_trials": 25,
        "seed": 17,
    }

    result_1 = run_parameter_recovery_from_config(config)
    result_2 = run_parameter_recovery_from_config(config)

    assert len(result_1.cases) == 4
    assert [case.true_params for case in result_1.cases] == [case.true_params for case in result_2.cases]
    for case in result_1.cases:
        assert 0.0 <= case.true_params["alpha"] <= 1.0
        assert 0.1 <= case.true_params["beta"] <= 5.0


def test_parameter_recovery_config_uses_shared_true_parameter_resolver(monkeypatch) -> None:
    """Config runner should delegate set/distribution resolution to shared helper."""

    captured: dict[str, object] = {}
    sentinel = object()

    def _fake_resolve_true_parameter_sets(
        *,
        true_parameter_sets=None,
        true_parameter_distributions=None,
        n_parameter_sets=None,
        seed=0,
    ):
        captured["sets"] = true_parameter_sets
        captured["dists"] = true_parameter_distributions
        captured["n_parameter_sets"] = n_parameter_sets
        captured["seed"] = seed
        return ({"alpha": 0.3, "beta": 2.0, "initial_value": 0.0},)

    def _fake_run_parameter_recovery(**kwargs):
        captured["resolved_sets"] = kwargs["true_parameter_sets"]
        return sentinel

    monkeypatch.setattr(
        "comp_model.recovery.config.resolve_true_parameter_sets_api",
        _fake_resolve_true_parameter_sets,
    )
    monkeypatch.setattr(
        "comp_model.recovery.config.run_parameter_recovery",
        _fake_run_parameter_recovery,
    )

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
            "type": "mle",
            "solver": "grid_search",
            "parameter_grid": {
                "alpha": [0.3],
                "beta": [2.0],
                "initial_value": [0.0],
            },
        },
        "true_parameter_distributions": {
            "alpha": {"distribution": "uniform", "lower": 0.0, "upper": 1.0},
            "beta": {"distribution": "uniform", "lower": 0.1, "upper": 5.0},
            "initial_value": {"distribution": "normal", "mean": 0.0, "std": 0.2},
        },
        "n_parameter_sets": 3,
        "n_trials": 25,
        "seed": 17,
    }

    result = run_parameter_recovery_from_config(config)

    assert result is sentinel
    assert captured["sets"] is None
    assert isinstance(captured["dists"], dict)
    assert captured["n_parameter_sets"] == 3
    assert captured["seed"] == 17
    assert captured["resolved_sets"] == ({"alpha": 0.3, "beta": 2.0, "initial_value": 0.0},)


def test_parameter_recovery_rejects_both_true_sets_and_distributions() -> None:
    """Config should fail when both explicit true sets and sampling distributions are provided."""

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
            "type": "mle", "solver": "grid_search",
            "parameter_grid": {
                "alpha": [0.3],
                "beta": [2.0],
                "initial_value": [0.0],
            },
        },
        "true_parameter_sets": [
            {"alpha": 0.3, "beta": 2.0, "initial_value": 0.0},
        ],
        "true_parameter_distributions": {
            "alpha": {"distribution": "uniform", "lower": 0.0, "upper": 1.0},
            "beta": {"distribution": "uniform", "lower": 0.1, "upper": 5.0},
            "initial_value": {"distribution": "normal", "mean": 0.0, "std": 0.2},
        },
        "n_parameter_sets": 2,
        "n_trials": 25,
        "seed": 17,
    }

    with pytest.raises(ValueError, match="exactly one"):
        run_parameter_recovery_from_config(config)


def test_parameter_recovery_sampling_requires_n_parameter_sets() -> None:
    """Sampling-based true parameter generation should require n_parameter_sets."""

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
            "type": "mle", "solver": "grid_search",
            "parameter_grid": {
                "alpha": [0.3],
                "beta": [2.0],
                "initial_value": [0.0],
            },
        },
        "true_parameter_distributions": {
            "alpha": {"distribution": "uniform", "lower": 0.0, "upper": 1.0},
            "beta": {"distribution": "uniform", "lower": 0.1, "upper": 5.0},
            "initial_value": {"distribution": "normal", "mean": 0.0, "std": 0.2},
        },
        "n_trials": 25,
        "seed": 17,
    }

    with pytest.raises(ValueError, match="n_parameter_sets is required"):
        run_parameter_recovery_from_config(config)


def test_parameter_recovery_supports_sampling_independent_param_mode() -> None:
    """Parameter recovery should support ``sampling`` independent param-space mode."""

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
            "type": "mle",
            "solver": "grid_search",
            "parameter_grid": {
                "alpha": [0.3],
                "beta": [2.0],
                "initial_value": [0.0],
            },
        },
        "sampling": {
            "mode": "independent",
            "space": "param",
            "n_parameter_sets": 3,
            "distributions": {
                "alpha": {"distribution": "uniform", "lower": 0.3, "upper": 0.3},
                "beta": {"distribution": "uniform", "lower": 2.0, "upper": 2.0},
                "initial_value": {"distribution": "uniform", "lower": 0.0, "upper": 0.0},
            },
        },
        "n_trials": 20,
        "seed": 18,
    }

    result = run_parameter_recovery_from_config(config)
    assert len(result.cases) == 3
    for case in result.cases:
        assert case.true_params == {"alpha": 0.3, "beta": 2.0, "initial_value": 0.0}


def test_parameter_recovery_supports_sampling_hierarchical_z_mode() -> None:
    """Parameter recovery should support hierarchical sampling in z-space."""

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
            "type": "mle",
            "solver": "grid_search",
            "parameter_grid": {
                "alpha": [0.5],
                "beta": [1.0],
                "initial_value": [0.0],
            },
        },
        "sampling": {
            "mode": "hierarchical",
            "space": "z",
            "n_parameter_sets": 4,
            "population": {
                "alpha": {"distribution": "uniform", "lower": 0.0, "upper": 0.0},
                "beta": {"distribution": "uniform", "lower": 0.0, "upper": 0.0},
                "initial_value": {"distribution": "uniform", "lower": 0.0, "upper": 0.0},
            },
            "individual_sd": {
                "alpha": 0.0,
                "beta": 0.0,
                "initial_value": 0.0,
            },
            "transforms": {
                "alpha": "unit_interval_logit",
                "beta": "positive_log",
                "initial_value": "identity",
            },
        },
        "n_trials": 20,
        "seed": 19,
    }

    result = run_parameter_recovery_from_config(config)
    assert len(result.cases) == 4
    for case in result.cases:
        assert case.true_params == {"alpha": 0.5, "beta": 1.0, "initial_value": 0.0}


def test_parameter_recovery_sampling_rejects_fixed_mode() -> None:
    """Sampling config should reject removed fixed mode."""

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
            "type": "mle",
            "solver": "grid_search",
            "parameter_grid": {
                "alpha": [0.3],
                "beta": [2.0],
                "initial_value": [0.0],
            },
        },
        "sampling": {
            "mode": "fixed",
            "space": "param",
            "n_parameter_sets": 1,
            "distributions": {
                "alpha": {"distribution": "uniform", "lower": 0.3, "upper": 0.3},
                "beta": {"distribution": "uniform", "lower": 2.0, "upper": 2.0},
                "initial_value": {"distribution": "uniform", "lower": 0.0, "upper": 0.0},
            },
        },
        "n_trials": 20,
        "seed": 20,
    }

    with pytest.raises(ValueError, match="fixed"):
        run_parameter_recovery_from_config(config)


def test_sampling_by_condition_emits_shared_delta_z_keys() -> None:
    """Condition-wise sampling should output wrapper-compatible shared+delta z keys."""

    sampled = _sample_true_parameter_sets_from_sampling(
        {
            "mode": "independent",
            "space": "param",
            "conditions": ["A", "B"],
            "baseline_condition": "A",
            "distributions": {
                "alpha": {"distribution": "uniform", "lower": 0.2, "upper": 0.2},
                "beta": {"distribution": "uniform", "lower": 1.0, "upper": 1.0},
            },
            "transforms": {
                "alpha": "unit_interval_logit",
                "beta": "positive_log",
            },
            "by_condition": {
                "B": {
                    "distributions": {
                        "alpha": {"distribution": "uniform", "lower": 0.8, "upper": 0.8},
                        "beta": {"distribution": "uniform", "lower": 2.0, "upper": 2.0},
                    },
                },
            },
        },
        n_parameter_sets=1,
        seed=21,
    )
    assert len(sampled) == 1
    params = sampled[0]
    assert set(params) == {
        "alpha__shared_z",
        "beta__shared_z",
        "alpha__delta_z__B",
        "beta__delta_z__B",
    }
    assert np.isclose(params["alpha__shared_z"], np.log(0.2 / 0.8))
    assert np.isclose(params["beta__shared_z"], np.log(1.0))
    assert np.isclose(
        params["alpha__delta_z__B"],
        np.log(0.8 / 0.2) - np.log(0.2 / 0.8),
    )
    assert np.isclose(params["beta__delta_z__B"], np.log(2.0))


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
                    "type": "mle", "solver": "grid_search",
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
                    "type": "mle", "solver": "grid_search",
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


def test_parameter_recovery_rejects_removed_scipy_map_config() -> None:
    """Parameter recovery should reject removed SciPy MAP estimator configs."""

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
            "initial_params": {"alpha": 0.4, "beta": 1.0, "initial_value": 0.0},
            "bounds": {
                "alpha": [0.0, 1.0],
                "beta": [0.0, 20.0],
                "initial_value": [None, None],
            },
        },
        "true_parameter_sets": [
            {"alpha": 0.3, "beta": 2.0, "initial_value": 0.0},
        ],
        "n_trials": 30,
        "seed": 13,
    }

    with pytest.raises(ValueError, match="estimator.type must be one of"):
        run_parameter_recovery_from_config(config)


def test_model_recovery_rejects_removed_scipy_map_candidates() -> None:
    """Model recovery config should reject removed SciPy MAP candidate definitions."""

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
                "name": "candidate_map",
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
                    "initial_params": {"alpha": 0.4, "beta": 1.0, "initial_value": 0.0},
                    "bounds": {
                        "alpha": [0.0, 1.0],
                        "beta": [0.0, 20.0],
                        "initial_value": [None, None],
                    },
                },
                "n_parameters": 3,
            },
            {
                "name": "candidate_grid",
                "model": {
                    "component_id": "asocial_state_q_value_softmax",
                    "kwargs": {},
                },
                "estimator": {
                    "type": "mle", "solver": "grid_search",
                    "parameter_grid": {
                        "alpha": [0.3],
                        "beta": [2.0],
                        "initial_value": [0.0],
                    },
                },
                "n_parameters": 3,
            },
        ],
        "n_trials": 30,
        "n_replications_per_generator": 2,
        "criterion": "log_likelihood",
        "seed": 21,
    }

    with pytest.raises(ValueError, match="estimator.type must be one of"):
        run_model_recovery_from_config(config)


def test_recovery_config_rejects_removed_scipy_map_estimators() -> None:
    """Recovery config should reject removed SciPy MAP estimator types."""

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
            "type": "scipy_map",
            "initial_params": {"alpha": 0.4, "beta": 1.0, "initial_value": 0.0},
            "bounds": {
                "alpha": [0.0, 1.0],
                "beta": [0.0, 20.0],
                "initial_value": [None, None],
            },
        },
        "true_parameter_sets": [
            {"alpha": 0.3, "beta": 2.0, "initial_value": 0.0},
        ],
        "n_trials": 10,
    }

    with pytest.raises(ValueError, match="estimator.type must be one of"):
        run_parameter_recovery_from_config(config)


def test_model_recovery_config_rejects_random_walk_candidate_estimators() -> None:
    """Model recovery config should reject removed random-walk estimators."""

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
                "name": "candidate_good_mcmc",
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
                    "type": "random_walk_metropolis",
                    "initial_params": {"alpha": 0.6, "beta": 3.0, "initial_value": 0.0},
                    "n_samples": 20,
                    "n_warmup": 20,
                    "thin": 1,
                    "proposal_scales": {"alpha": 0.05, "beta": 0.2, "initial_value": 0.1},
                    "bounds": {
                        "alpha": [0.0, 1.0],
                        "beta": [0.0, 20.0],
                        "initial_value": [None, None],
                    },
                    "random_seed": 5,
                },
                "n_parameters": 3,
            },
            {
                "name": "candidate_bad_mcmc",
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
                    "type": "random_walk_metropolis",
                    "initial_params": {"alpha": 0.2, "beta": 0.1, "initial_value": 0.0},
                    "n_samples": 20,
                    "n_warmup": 20,
                    "thin": 1,
                    "proposal_scales": {"alpha": 0.02, "beta": 0.05, "initial_value": 0.05},
                    "bounds": {
                        "alpha": [0.0, 0.4],
                        "beta": [0.0, 0.5],
                        "initial_value": [None, None],
                    },
                    "random_seed": 6,
                },
                "n_parameters": 3,
            },
        ],
        "n_trials": 30,
        "n_replications_per_generator": 1,
        "criterion": "waic",
        "seed": 22,
    }

    with pytest.raises(ValueError, match="estimator.type must be one of"):
        run_model_recovery_from_config(config)


def test_parameter_recovery_supports_likelihood_config() -> None:
    """Parameter recovery config should accept explicit likelihood sections."""

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
            "type": "mle", "solver": "grid_search",
            "parameter_grid": {
                "alpha": [0.3],
                "beta": [2.0],
                "initial_value": [0.0],
            },
        },
        "likelihood": {
            "type": "actor_subset_replay",
            "fitted_actor_id": "subject",
            "scored_actor_ids": ["subject"],
            "auto_fill_unmodeled_actors": True,
        },
        "true_parameter_sets": [
            {"alpha": 0.3, "beta": 2.0, "initial_value": 0.0},
        ],
        "n_trials": 20,
        "seed": 33,
    }

    result = run_parameter_recovery_from_config(config)
    assert len(result.cases) == 1


def test_model_recovery_config_rejects_invalid_candidate_likelihood() -> None:
    """Model recovery should fail fast on invalid candidate likelihood config."""

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
                "name": "candidate_bad_likelihood",
                "model": {
                    "component_id": "asocial_state_q_value_softmax",
                    "kwargs": {},
                },
                "estimator": {
                    "type": "mle", "solver": "grid_search",
                    "parameter_grid": {
                        "alpha": [0.3],
                        "beta": [2.0],
                        "initial_value": [0.0],
                    },
                },
                "likelihood": {"type": "not_supported"},
                "n_parameters": 3,
            },
        ],
        "n_trials": 10,
        "n_replications_per_generator": 1,
        "seed": 9,
    }

    with pytest.raises(ValueError, match="likelihood.type must be one of"):
        run_model_recovery_from_config(config)


def test_parameter_recovery_supports_generator_social_simulation_config() -> None:
    """Parameter recovery config should support social generator simulation."""

    n_trials = 60
    config = {
        "simulation": {
            "type": "generator",
            "generator": {
                "component_id": "event_trace_social_pre_choice_generator",
                "kwargs": {},
            },
            "demonstrator_model": {
                "component_id": "fixed_sequence_demonstrator",
                "kwargs": {"sequence": [1] * n_trials},
            },
            "block": {
                "n_trials": n_trials,
                "program_kwargs": {"reward_probabilities": [0.5, 0.5]},
            },
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
            "type": "mle", "solver": "grid_search",
            "parameter_grid": {
                "alpha": [0.3],
                "beta": [4.0],
                "initial_value": [0.0],
            },
        },
        "likelihood": {
            "type": "actor_subset_replay",
            "fitted_actor_id": "subject",
            "scored_actor_ids": ["subject"],
            "auto_fill_unmodeled_actors": True,
        },
        "true_parameter_sets": [
            {"alpha": 0.3, "beta": 4.0, "initial_value": 0.0},
        ],
        "n_trials": n_trials,
        "seed": 44,
    }

    result = run_parameter_recovery_from_config(config)
    assert len(result.cases) == 1
    assert set(result.cases[0].estimated_params) == {"alpha", "beta", "initial_value"}


def test_model_recovery_supports_generator_social_simulation_config() -> None:
    """Model recovery config should support social generator simulation."""

    n_trials = 80
    config = {
        "simulation": {
            "type": "generator",
            "generator": {
                "component_id": "event_trace_social_pre_choice_generator",
                "kwargs": {},
            },
            "demonstrator_model": {
                "component_id": "fixed_sequence_demonstrator",
                "kwargs": {"sequence": [1] * n_trials},
            },
            "block": {
                "n_trials": n_trials,
                "program_kwargs": {"reward_probabilities": [0.5, 0.5]},
            },
        },
        "generating": [
            {
                "name": "qrl_generator",
                "model": {
                    "component_id": "asocial_state_q_value_softmax",
                    "kwargs": {},
                },
                "true_params": {"alpha": 0.3, "beta": 4.0, "initial_value": 0.0},
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
                    "type": "mle", "solver": "grid_search",
                    "parameter_grid": {
                        "alpha": [0.3],
                        "beta": [4.0],
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
                    "type": "mle", "solver": "grid_search",
                    "parameter_grid": {
                        "alpha": [0.8],
                        "beta": [0.1],
                        "initial_value": [1.0],
                    },
                },
                "n_parameters": 3,
            },
        ],
        "likelihood": {
            "type": "actor_subset_replay",
            "fitted_actor_id": "subject",
            "scored_actor_ids": ["subject"],
            "auto_fill_unmodeled_actors": True,
        },
        "n_trials": n_trials,
        "n_replications_per_generator": 2,
        "criterion": "log_likelihood",
        "seed": 45,
    }

    result = run_model_recovery_from_config(config)
    assert len(result.cases) == 2
    assert sum(result.confusion_matrix["qrl_generator"].values()) == 2


def test_generator_social_simulation_requires_demonstrator_model() -> None:
    """Social generator simulation config should require demonstrator model."""

    config = {
        "simulation": {
            "type": "generator",
            "generator": {
                "component_id": "event_trace_social_pre_choice_generator",
                "kwargs": {},
            },
            "block": {
                "n_trials": 10,
                "program_kwargs": {"reward_probabilities": [0.5, 0.5]},
            },
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
            "type": "mle", "solver": "grid_search",
            "parameter_grid": {
                "alpha": [0.3],
                "beta": [4.0],
                "initial_value": [0.0],
            },
        },
        "true_parameter_sets": [
            {"alpha": 0.3, "beta": 4.0, "initial_value": 0.0},
        ],
        "n_trials": 10,
        "seed": 46,
    }

    with pytest.raises(ValueError, match="simulation.demonstrator_model"):
        run_parameter_recovery_from_config(config)


def test_parameter_recovery_config_rejects_unknown_top_level_keys() -> None:
    """Parameter recovery config should fail fast on unknown top-level keys."""

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
            "type": "mle", "solver": "grid_search",
            "parameter_grid": {
                "alpha": [0.3],
                "beta": [2.0],
                "initial_value": [0.0],
            },
        },
        "true_parameter_sets": [
            {"alpha": 0.3, "beta": 2.0, "initial_value": 0.0},
        ],
        "n_trials": 10,
        "unexpected": "typo",
    }

    with pytest.raises(ValueError, match="config has unknown keys"):
        run_parameter_recovery_from_config(config)


def test_generator_simulation_block_rejects_unknown_keys() -> None:
    """Generator simulation block config should reject unknown keys."""

    config = {
        "simulation": {
            "type": "generator",
            "generator": {
                "component_id": "event_trace_asocial_generator",
                "kwargs": {},
            },
            "block": {
                "n_trials": 10,
                "problem_kwargs": {"reward_probabilities": [0.2, 0.8]},
                "bad_field": 1,
            },
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
            "type": "mle", "solver": "grid_search",
            "parameter_grid": {
                "alpha": [0.3],
                "beta": [2.0],
                "initial_value": [0.0],
            },
        },
        "true_parameter_sets": [
            {"alpha": 0.3, "beta": 2.0, "initial_value": 0.0},
        ],
        "n_trials": 10,
        "seed": 52,
    }

    with pytest.raises(ValueError, match="simulation.blocks\\[0\\] has unknown keys"):
        run_parameter_recovery_from_config(config)


def test_parameter_recovery_supports_subject_level_joint_block_fit_strategy() -> None:
    """Parameter recovery should support one-parameter-set subject-level fitting."""

    config = {
        "simulation": {
            "type": "generator",
            "level": "subject",
            "generator": {
                "component_id": "event_trace_asocial_generator",
                "kwargs": {},
            },
            "block": {
                "n_trials": 10,
                "problem_kwargs": {"reward_probabilities": [0.2, 0.8]},
            },
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
            "type": "mle", "solver": "grid_search",
            "parameter_grid": {
                "alpha": [0.3],
                "beta": [2.0],
                "initial_value": [0.0],
            },
        },
        "block_fit_strategy": "joint",
        "true_parameter_sets": [
            {"alpha": 0.3, "beta": 2.0, "initial_value": 0.0},
        ],
        "n_trials": 10,
        "seed": 54,
    }

    result = run_parameter_recovery_from_config(config)
    assert len(result.cases) == 1
    assert result.cases[0].estimated_params == {
        "alpha": 0.3,
        "beta": 2.0,
        "initial_value": 0.0,
    }


def test_parameter_recovery_rejects_subject_level_independent_block_fit_strategy() -> None:
    """Subject-level parameter recovery should reject block-wise MLE estimates."""

    config = {
        "simulation": {
            "type": "generator",
            "level": "subject",
            "generator": {
                "component_id": "event_trace_asocial_generator",
                "kwargs": {},
            },
            "block": {
                "n_trials": 10,
                "problem_kwargs": {"reward_probabilities": [0.2, 0.8]},
            },
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
            "type": "mle",
            "solver": "grid_search",
            "parameter_grid": {
                "alpha": [0.3],
                "beta": [2.0],
                "initial_value": [0.0],
            },
        },
        "true_parameter_sets": [
            {"alpha": 0.3, "beta": 2.0, "initial_value": 0.0},
        ],
        "n_trials": 10,
        "seed": 54,
    }

    with pytest.raises(
        ValueError,
        match="subject-level parameter recovery requires one shared parameter estimate per subject",
    ):
        run_parameter_recovery_from_config(config)


def test_parameter_recovery_supports_subject_level_shared_stan_estimator(monkeypatch) -> None:
    """Subject-level parameter recovery should dispatch shared-parameter Stan estimators."""

    captured: dict[str, object] = {}

    class _FakeMapCandidate:
        def __init__(self) -> None:
            self.params = {"alpha": 0.25, "beta": 1.5, "initial_value": 0.0}
            self.log_likelihood = -10.0
            self.log_posterior = -10.5

    class _FakePosteriorResult:
        def __init__(self) -> None:
            self.map_candidate = _FakeMapCandidate()

    def _fake_fit_subject_auto_from_config(subject, *, config, registry=None):
        del registry
        captured["n_blocks"] = len(subject.blocks)
        captured["estimator_type"] = config["estimator"]["type"]
        return _FakePosteriorResult()

    monkeypatch.setattr(
        "comp_model.recovery.config.fit_subject_auto_from_config",
        _fake_fit_subject_auto_from_config,
    )

    config = {
        "simulation": {
            "type": "generator",
            "level": "subject",
            "generator": {
                "component_id": "event_trace_asocial_generator",
                "kwargs": {},
            },
            "blocks": [
                {
                    "n_trials": 12,
                    "problem_kwargs": {"reward_probabilities": [0.2, 0.8]},
                    "block_id": "b1",
                },
                {
                    "n_trials": 12,
                    "problem_kwargs": {"reward_probabilities": [0.2, 0.8]},
                    "block_id": "b2",
                },
            ],
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
            "type": "subject_shared_stan_map",
            "parameter_names": ["alpha", "beta", "initial_value"],
        },
        "true_parameter_sets": [
            {"alpha": 0.3, "beta": 2.0, "initial_value": 0.0},
        ],
        "n_trials": 12,
        "seed": 57,
    }

    result = run_parameter_recovery_from_config(config)
    assert len(result.cases) == 1
    assert result.cases[0].estimated_params == {
        "alpha": 0.25,
        "beta": 1.5,
        "initial_value": 0.0,
    }
    assert captured["estimator_type"] == "subject_shared_stan_map"
    assert captured["n_blocks"] == 2


def test_parameter_recovery_rejects_study_level_simulation() -> None:
    """Parameter recovery should reject study-level simulation data."""

    config = {
        "simulation": {
            "type": "generator",
            "level": "study",
            "n_subjects": 2,
            "generator": {
                "component_id": "event_trace_asocial_generator",
                "kwargs": {},
            },
            "block": {
                "n_trials": 10,
                "problem_kwargs": {"reward_probabilities": [0.2, 0.8]},
            },
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
            "type": "mle", "solver": "grid_search",
            "parameter_grid": {
                "alpha": [0.3],
                "beta": [2.0],
                "initial_value": [0.0],
            },
        },
        "true_parameter_sets": [
            {"alpha": 0.3, "beta": 2.0, "initial_value": 0.0},
        ],
        "n_trials": 10,
        "seed": 58,
    }

    with pytest.raises(ValueError, match="supports simulation.level in \\{'block', 'subject'\\}"):
        run_parameter_recovery_from_config(config)


def test_model_recovery_supports_study_level_generator_simulation() -> None:
    """Model recovery should support study-level generator simulation configs."""

    config = {
        "simulation": {
            "type": "generator",
            "level": "study",
            "n_subjects": 2,
            "generator": {
                "component_id": "event_trace_asocial_generator",
                "kwargs": {},
            },
            "blocks": [
                {
                    "n_trials": 30,
                    "problem_kwargs": {"reward_probabilities": [0.2, 0.8]},
                    "block_id": "b1",
                },
                {
                    "n_trials": 30,
                    "problem_kwargs": {"reward_probabilities": [0.2, 0.8]},
                    "block_id": "b2",
                },
            ],
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
                    "type": "mle", "solver": "grid_search",
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
                    "type": "mle", "solver": "grid_search",
                    "parameter_grid": {
                        "alpha": [0.9],
                        "beta": [0.1],
                        "initial_value": [1.0],
                    },
                },
                "n_parameters": 3,
            },
        ],
        "n_trials": 30,
        "n_replications_per_generator": 2,
        "criterion": "log_likelihood",
        "seed": 55,
    }

    result = run_model_recovery_from_config(config)
    assert len(result.cases) == 2
    assert sum(result.confusion_matrix["qrl_generator"].values()) == 2


def test_model_recovery_supports_joint_block_fit_strategy() -> None:
    """Model recovery config should support joint subject-level block fitting."""

    config = {
        "simulation": {
            "type": "generator",
            "level": "study",
            "n_subjects": 2,
            "generator": {
                "component_id": "event_trace_asocial_generator",
                "kwargs": {},
            },
            "blocks": [
                {
                    "n_trials": 25,
                    "problem_kwargs": {"reward_probabilities": [0.2, 0.8]},
                    "block_id": "b1",
                },
                {
                    "n_trials": 25,
                    "problem_kwargs": {"reward_probabilities": [0.2, 0.8]},
                    "block_id": "b2",
                },
            ],
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
                    "type": "mle", "solver": "grid_search",
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
                    "type": "mle", "solver": "grid_search",
                    "parameter_grid": {
                        "alpha": [0.9],
                        "beta": [0.1],
                        "initial_value": [1.0],
                    },
                },
                "n_parameters": 3,
            },
        ],
        "block_fit_strategy": "joint",
        "n_trials": 25,
        "n_replications_per_generator": 2,
        "criterion": "log_likelihood",
        "seed": 57,
    }

    result = run_model_recovery_from_config(config)
    assert len(result.cases) == 2
    assert sum(result.confusion_matrix["qrl_generator"].values()) == 2
