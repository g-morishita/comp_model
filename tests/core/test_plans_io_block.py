import builtins
import json
from pathlib import Path

import pytest

import comp_model_core.plans as plans
from comp_model_core.plans.block import BlockPlan, StudyPlan
from comp_model_core.plans.io import (
    _deep_merge,
    _expand_trial_specs,
    _normalize_block_dict,
    _expand_subjects,
    study_plan_from_dict,
    load_study_plan_yaml,
    infer_load_conditions,
)


TRIAL_TEMPLATE = {"self_outcome": {"kind": "VERIDICAL"}, "available_actions": [0, 1]}


def _base_block(block_id="b1"):
    return {
        "block_id": block_id,
        "condition": "c1",
        "n_trials": 2,
        "bandit_type": "bernoulli",
        "bandit_config": {"probs": [0.2, 0.8]},
        "trial_spec_template": dict(TRIAL_TEMPLATE),
    }


def test_plans_init_exports():
    assert plans.BlockPlan is BlockPlan
    assert plans.StudyPlan is StudyPlan
    assert "infer_load_conditions" in plans.__all__


def test_blockplan_validation_errors():
    base_specs = [dict(TRIAL_TEMPLATE) for _ in range(2)]

    with pytest.raises(ValueError):
        BlockPlan(
            block_id="b1",
            condition="",
            n_trials=2,
            bandit_type="bernoulli",
            bandit_config={"probs": [0.2, 0.8]},
            trial_specs=base_specs,
        )

    with pytest.raises(ValueError):
        BlockPlan(
            block_id="b1",
            condition="c",
            n_trials=2,
            bandit_type="",
            bandit_config={"probs": [0.2, 0.8]},
            trial_specs=base_specs,
        )

    with pytest.raises(ValueError):
        BlockPlan(
            block_id="b1",
            condition="c",
            n_trials=2,
            bandit_type="bernoulli",
            bandit_config=None,
            trial_specs=base_specs,
        )

    with pytest.raises(ValueError):
        BlockPlan(
            block_id="b1",
            condition="c",
            n_trials=2,
            bandit_type="bernoulli",
            bandit_config={"probs": [0.2, 0.8]},
            trial_specs="bad",
        )

    with pytest.raises(ValueError):
        BlockPlan(
            block_id="b1",
            condition="c",
            n_trials=2,
            bandit_type="bernoulli",
            bandit_config={"probs": [0.2, 0.8]},
            trial_specs=[dict(TRIAL_TEMPLATE)],
        )

    with pytest.raises(ValueError):
        BlockPlan(
            block_id="b1",
            condition="c",
            n_trials=2,
            bandit_type="bernoulli",
            bandit_config={"probs": [0.2, 0.8]},
            trial_specs=[dict(TRIAL_TEMPLATE), "nope"],
        )


def test_studyplan_requires_subjects():
    with pytest.raises(ValueError):
        StudyPlan(subjects={})


def test_deep_merge_and_expand_trial_specs_success():
    merged = _deep_merge({"a": {"x": 1}, "b": [1, 2]}, {"a": {"y": 2}, "b": [9]})
    assert merged == {"a": {"x": 1, "y": 2}, "b": [9]}

    block = {
        "n_trials": 3,
        "trial_spec_template": {"self_outcome": {"kind": "VERIDICAL"}, "meta": {"x": 1}},
        "trial_spec_overrides": [None, {"meta": {"y": 2}, "available_actions": [1]}, None],
    }
    specs = _expand_trial_specs(block)
    assert specs[0]["meta"] == {"x": 1}
    assert specs[1]["meta"] == {"x": 1, "y": 2}
    assert specs[1]["available_actions"] == [1]

    raw_block = {
        "n_trials": 2,
        "trial_specs": [dict(TRIAL_TEMPLATE), dict(TRIAL_TEMPLATE)],
    }
    assert _expand_trial_specs(raw_block) == raw_block["trial_specs"]


def test_expand_trial_specs_errors():
    with pytest.raises(ValueError, match="missing required key"):
        _expand_trial_specs({})

    with pytest.raises(ValueError, match="n_trials must be > 0"):
        _expand_trial_specs({"n_trials": 0, "trial_specs": []})

    with pytest.raises(ValueError, match="Specify either 'trial_specs'"):
        _expand_trial_specs({"n_trials": 1, "trial_specs": [{}], "trial_spec_template": {}})

    with pytest.raises(ValueError, match="trial_specs must be a list"):
        _expand_trial_specs({"n_trials": 1, "trial_specs": "bad"})

    with pytest.raises(ValueError, match="trial_specs length must equal n_trials"):
        _expand_trial_specs({"n_trials": 2, "trial_specs": [{}]})

    with pytest.raises(ValueError, match="trial_specs\\[0\\] must be a dict"):
        _expand_trial_specs({"n_trials": 1, "trial_specs": ["bad"]})

    with pytest.raises(ValueError, match="Missing required trial schedule"):
        _expand_trial_specs({"n_trials": 1})

    with pytest.raises(ValueError, match="trial_spec_template must be a mapping"):
        _expand_trial_specs({"n_trials": 1, "trial_spec_template": "bad"})

    with pytest.raises(ValueError, match="trial_spec_overrides must be a list"):
        _expand_trial_specs(
            {"n_trials": 1, "trial_spec_template": dict(TRIAL_TEMPLATE), "trial_spec_overrides": "bad"}
        )

    with pytest.raises(ValueError, match="trial_spec_overrides length must equal n_trials"):
        _expand_trial_specs(
            {"n_trials": 2, "trial_spec_template": dict(TRIAL_TEMPLATE), "trial_spec_overrides": [None]}
        )

    with pytest.raises(ValueError, match="trial_spec_overrides\\[0\\] must be a mapping"):
        _expand_trial_specs(
            {"n_trials": 1, "trial_spec_template": dict(TRIAL_TEMPLATE), "trial_spec_overrides": ["bad"]}
        )


def test_normalize_block_dict_removes_template_keys():
    block = _base_block()
    block["trial_spec_overrides"] = [None, {"available_actions": [1]}]
    out = _normalize_block_dict(block)
    assert "trial_spec_template" not in out
    assert "trial_spec_overrides" not in out
    assert out["trial_specs"][1]["available_actions"] == [1]

    with pytest.raises(ValueError, match="Each block must be a mapping"):
        _normalize_block_dict("bad")


def test_expand_subjects_variants_and_errors():
    base = _base_block(block_id="b_{subject}_{rep}")
    base["repeat"] = 2

    out = _expand_subjects({"subjects": 3, "blocks_template": [base]})
    assert list(out.keys()) == ["s01", "s02", "s03"]
    assert out["s01"][0]["block_id"] == "b_s01_1"
    assert out["s01"][1]["block_id"] == "b_s01_2"
    assert "repeat" not in out["s01"][0]

    base2 = _base_block(block_id="b")
    base2["repeat"] = 2
    out2 = _expand_subjects({"subjects": ["A"], "blocks_template": [base2]})
    assert out2["A"][0]["block_id"] == "b_r1"
    assert out2["A"][1]["block_id"] == "b_r2"

    out3 = _expand_subjects({"subjects": ["S1"], "subject_template": {"blocks": [_base_block()]}})
    assert "S1" in out3

    out4 = _expand_subjects({"subjects": {"S1": []}})
    assert out4 == {"S1": []}

    with pytest.raises(ValueError, match="subjects integer must be > 0"):
        _expand_subjects({"subjects": 0, "blocks_template": [_base_block()]})

    with pytest.raises(ValueError, match="Specify exactly one of 'blocks_template' or 'subject_template'"):
        _expand_subjects({"subjects": ["S1"]})

    with pytest.raises(ValueError, match="subject_template.blocks"):
        _expand_subjects({"subjects": ["S1"], "subject_template": {}})

    with pytest.raises(ValueError, match="Specify exactly one of 'blocks_template' or 'subject_template'"):
        _expand_subjects(
            {
                "subjects": ["S1"],
                "blocks_template": [_base_block()],
                "subject_template": {"blocks": [_base_block()]},
            }
        )

    with pytest.raises(ValueError, match="subjects list must contain"):
        _expand_subjects({"subjects": [object()], "blocks_template": [_base_block()]})

    with pytest.raises(ValueError, match="Each template block must be a mapping"):
        _expand_subjects({"subjects": ["S1"], "blocks_template": ["bad"]})

    bad_repeat = _base_block()
    bad_repeat["repeat"] = "x"
    with pytest.raises(ValueError, match="repeat must be an integer"):
        _expand_subjects({"subjects": ["S1"], "blocks_template": [bad_repeat]})

    bad_repeat2 = _base_block()
    bad_repeat2["repeat"] = 0
    with pytest.raises(ValueError, match="repeat must be > 0"):
        _expand_subjects({"subjects": ["S1"], "blocks_template": [bad_repeat2]})

    with pytest.raises(ValueError, match="Input must contain key 'subjects'"):
        _expand_subjects({"subjects": "bad"})


def test_study_plan_from_dict_metadata_and_errors():
    with pytest.raises(ValueError, match="must contain key 'subjects'"):
        study_plan_from_dict({})

    with pytest.raises(ValueError, match="subjects\\[S1\\] must be a list"):
        study_plan_from_dict({"subjects": {"S1": "bad"}})

    raw = {"subjects": {"S1": [_base_block()]}, "metadata": "not-a-dict"}
    plan = study_plan_from_dict(raw)
    assert plan.metadata == {}
    assert plan.subjects["S1"][0].n_trials == 2


def test_load_study_plan_yaml_import_error(monkeypatch, tmp_path: Path):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("no yaml")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="PyYAML is required"):
        load_study_plan_yaml(str(tmp_path / "plan.yaml"))


def test_load_study_plan_yaml_non_mapping(tmp_path: Path):
    yaml = pytest.importorskip("yaml")
    p = tmp_path / "plan.yaml"
    p.write_text(yaml.safe_dump([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match="YAML root must be a mapping"):
        load_study_plan_yaml(str(p))


def test_infer_load_conditions_variants(tmp_path: Path, monkeypatch):
    raw = {
        "subjects": {
            "S1": [
                {
                    "block_id": "b1",
                    "condition": "c1",
                    "n_trials": 1,
                    "bandit_type": "bernoulli",
                    "bandit_config": {"probs": [0.2, 0.8]},
                    "trial_spec_template": dict(TRIAL_TEMPLATE),
                },
                {
                    "block_id": "b2",
                    "condition": "c2",
                    "n_trials": 1,
                    "bandit_type": "bernoulli",
                    "bandit_config": {"probs": [0.2, 0.8]},
                    "trial_spec_template": dict(TRIAL_TEMPLATE),
                },
            ],
            "S2": [
                {
                    "block_id": "b3",
                    "condition": "c2",
                    "n_trials": 1,
                    "bandit_type": "bernoulli",
                    "bandit_config": {"probs": [0.2, 0.8]},
                    "trial_spec_template": dict(TRIAL_TEMPLATE),
                },
                {
                    "block_id": "b4",
                    "condition": "c3",
                    "n_trials": 1,
                    "bandit_type": "bernoulli",
                    "bandit_config": {"probs": [0.2, 0.8]},
                    "trial_spec_template": dict(TRIAL_TEMPLATE),
                },
            ],
        }
    }
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    assert infer_load_conditions(str(p)) == ["c1", "c2", "c3"]

    p2 = tmp_path / "plan.txt"
    p2.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="\\.yaml/.yml or \\.json"):
        infer_load_conditions(str(p2))

    p3 = tmp_path / "empty.json"
    p3.write_text(json.dumps({"subjects": {"S1": []}}), encoding="utf-8")
    with pytest.raises(ValueError, match="No conditions found"):
        infer_load_conditions(str(p3))

    class BadBlock:
        condition = ""

        def __repr__(self) -> str:  # pragma: no cover - for error messages only
            return "BadBlock"

    class BadPlan:
        subjects = {"S1": [BadBlock()]}

    monkeypatch.setattr("comp_model_core.plans.io.load_study_plan_json", lambda _: BadPlan())
    p4 = tmp_path / "bad.json"
    p4.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required condition"):
        infer_load_conditions(str(p4))
