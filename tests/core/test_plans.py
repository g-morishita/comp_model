import json
from pathlib import Path

import pytest

from comp_model_core.plans.block import BlockPlan
from comp_model_core.plans.io import study_plan_from_dict, load_study_plan_json, load_study_plan


def test_blockplan_validation_demonstrator_pairing():
    # only one of demonstrator_type / demonstrator_config -> error
    with pytest.raises(ValueError):
        BlockPlan(
            block_id="b1",
            condition="c",
            n_trials=10,
            bandit_type="bernoulli",
            bandit_config={"probs": [0.2, 0.8]},
            trial_specs=[{"self_outcome": {"kind": "VERIDICAL"}, "available_actions": [0, 1]} for _ in range(10)],
            demonstrator_type="noisy_best",
            demonstrator_config=None,
        )


def test_blockplan_validation_basic_fields():
    with pytest.raises(ValueError):
        BlockPlan(
            block_id="",
            condition="c",
            n_trials=10,
            bandit_type="bernoulli",
            bandit_config={"probs": [0.2, 0.8]},
            trial_specs=[{"self_outcome": {"kind": "VERIDICAL"}, "available_actions": [0, 1]} for _ in range(10)],
        )

    with pytest.raises(ValueError):
        BlockPlan(
            block_id="b1",
            condition="c",
            n_trials=0,
            bandit_type="bernoulli",
            bandit_config={"probs": [0.2, 0.8]},
            trial_specs=[],
        )


def test_study_plan_from_dict_minimal():
    raw = {
        "subjects": {
            "S1": [
                {
                    "block_id": "b1",
                    "n_trials": 3,
                    "condition": "c",
                    "bandit_type": "bernoulli",
                    "bandit_config": {"probs": [0.2, 0.8]},
                    "trial_spec_template": {"self_outcome": {"kind": "VERIDICAL"}, "available_actions": [0, 1]},
                }
            ]
        }
    }
    plan = study_plan_from_dict(raw)
    assert "S1" in plan.subjects
    assert len(plan.subjects["S1"]) == 1
    assert plan.subjects["S1"][0].block_id == "b1"


def test_load_study_plan_json(tmp_path: Path):
    raw = {
        "subjects": {
            "S1": [
                {
                    "block_id": "b1",
                    "condition": "c",
                    "n_trials": 3,
                    "bandit_type": "bernoulli",
                    "bandit_config": {"probs": [0.2, 0.8]},
                    "trial_spec_template": {"self_outcome": {"kind": "VERIDICAL"}, "available_actions": [0, 1]},
                }
            ]
        }
    }
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    plan = load_study_plan_json(str(p))
    assert "S1" in plan.subjects


def test_load_study_plan_yaml_optional(tmp_path: Path):
    yaml = pytest.importorskip("yaml")  # skip if PyYAML not installed

    from comp_model_core.plans.io import load_study_plan_yaml

    raw = {
        "subjects": {
            "S1": [
                {
                    "block_id": "b1",
                    "condition": "c",
                    "n_trials": 3,
                    "bandit_type": "bernoulli",
                    "bandit_config": {"probs": [0.2, 0.8]},
                    "trial_spec_template": {"self_outcome": {"kind": "VERIDICAL"}, "available_actions": [0, 1]},
                }
            ]
        }
    }
    p = tmp_path / "plan.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    plan = load_study_plan_yaml(str(p))
    assert "S1" in plan.subjects


def test_load_study_plan_dispatch_json(tmp_path: Path):
    raw = {
        "subjects": {
            "S1": [
                {
                    "block_id": "b1",
                    "condition": "c",
                    "n_trials": 3,
                    "bandit_type": "bernoulli",
                    "bandit_config": {"probs": [0.2, 0.8]},
                    "trial_spec_template": {"self_outcome": {"kind": "VERIDICAL"}, "available_actions": [0, 1]},
                }
            ]
        }
    }
    # A case of a JSON file
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    plan = load_study_plan(str(p))
    assert "S1" in plan.subjects


def test_load_study_plan_dispatch_yaml_optional(tmp_path: Path):
    yaml = pytest.importorskip("yaml")  # skip if PyYAML not installed
    raw = {
        "subjects": {
            "S1": [
                {
                    "block_id": "b1",
                    "condition": "c",
                    "n_trials": 3,
                    "bandit_type": "bernoulli",
                    "bandit_config": {"probs": [0.2, 0.8]},
                    "trial_spec_template": {"self_outcome": {"kind": "VERIDICAL"}, "available_actions": [0, 1]},
                }
            ]
        }
    }

    # A case of a YAML file.
    p = tmp_path / "plan.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    plan = load_study_plan(str(p))
    assert "S1" in plan.subjects


def test_load_study_plan_raises_value_error_for_unsupported_extension(tmp_path: Path):
    raw = {
        "subjects": {
            "S1": [
                {
                    "block_id": "b1",
                    "condition": "c",
                    "n_trials": 3,
                    "bandit_type": "bernoulli",
                    "bandit_config": {"probs": [0.2, 0.8]},
                    "trial_spec_template": {"self_outcome": {"kind": "VERIDICAL"}, "available_actions": [0, 1]},
                }
            ]
        }
    }

    p = tmp_path / "plan.text"
    p.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="\\.yaml/.yml or \\.json"):
        load_study_plan(str(p))