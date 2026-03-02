"""Tests for config-driven recovery CLI."""

from __future__ import annotations

import json

from comp_model.recovery import run_recovery_cli


def test_recovery_cli_parameter_mode_writes_outputs(tmp_path, capsys) -> None:
    """Parameter mode should write case CSV and summary JSON files."""

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
        "n_trials": 6,
        "seed": 123,
    }
    config_path = tmp_path / "parameter_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    code = run_recovery_cli(
        [
            "--config",
            str(config_path),
            "--mode",
            "parameter",
            "--output-dir",
            str(tmp_path),
            "--prefix",
            "param",
        ]
    )
    captured = capsys.readouterr()

    assert code == 0
    assert "Parameter recovery complete" in captured.out
    assert (tmp_path / "param_parameter_cases.csv").exists()
    assert (tmp_path / "param_parameter_summary.json").exists()


def test_recovery_cli_model_mode_writes_outputs(tmp_path, capsys) -> None:
    """Model mode should write model-case, confusion, and summary files."""

    grid_estimator = {
        "type": "mle", "solver": "grid_search",
        "parameter_grid": {
            "alpha": [0.3],
            "beta": [2.0],
            "initial_value": [0.0],
        },
    }
    config = {
        "problem": {
            "component_id": "stationary_bandit",
            "kwargs": {"reward_probabilities": [0.2, 0.8]},
        },
        "generating": [
            {
                "name": "gen_q",
                "model": {
                    "component_id": "asocial_state_q_value_softmax",
                    "kwargs": {},
                },
                "true_params": {"alpha": 0.3, "beta": 2.0, "initial_value": 0.0},
            }
        ],
        "candidates": [
            {
                "name": "cand_q",
                "model": {
                    "component_id": "asocial_state_q_value_softmax",
                    "kwargs": {},
                },
                "estimator": grid_estimator,
            }
        ],
        "n_trials": 6,
        "n_replications_per_generator": 1,
        "seed": 123,
    }
    config_path = tmp_path / "model_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    code = run_recovery_cli(
        [
            "--config",
            str(config_path),
            "--mode",
            "model",
            "--output-dir",
            str(tmp_path),
            "--prefix",
            "model",
        ]
    )
    captured = capsys.readouterr()

    assert code == 0
    assert "Model recovery complete" in captured.out
    assert (tmp_path / "model_model_cases.csv").exists()
    assert (tmp_path / "model_model_confusion.csv").exists()
    assert (tmp_path / "model_model_summary.json").exists()


def test_recovery_cli_auto_mode_infers_parameter_from_config(tmp_path) -> None:
    """Auto mode should infer parameter recovery from parameter keys."""

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
        "n_trials": 4,
    }
    config_path = tmp_path / "auto_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    code = run_recovery_cli(
        [
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path),
            "--prefix",
            "auto",
        ]
    )

    assert code == 0
    assert (tmp_path / "auto_parameter_cases.csv").exists()


def test_recovery_cli_accepts_yaml_config(tmp_path, capsys) -> None:
    """Recovery CLI should parse YAML configs."""

    config_path = tmp_path / "parameter_config.yaml"
    config_path.write_text(
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
                "n_trials: 6",
                "seed: 123",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    code = run_recovery_cli(
        [
            "--config",
            str(config_path),
            "--mode",
            "parameter",
            "--output-dir",
            str(tmp_path),
            "--prefix",
            "param_yaml",
        ]
    )
    captured = capsys.readouterr()

    assert code == 0
    assert "Parameter recovery complete" in captured.out
    assert (tmp_path / "param_yaml_parameter_cases.csv").exists()
