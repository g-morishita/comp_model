"""Tests for model recovery configuration parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from comp_model_impl.recovery.model.config import (
    CandidateModelSpec,
    GeneratingModelSpec,
    ModelRecoveryConfig,
    _coerce_component_ref,
    _coerce_float_mapping,
    _coerce_mapping,
    _coerce_sampling_spec,
    config_from_raw_dict,
    config_to_json,
    load_model_recovery_config,
)


def _minimal_raw_config() -> dict:
    """Build a minimal valid raw config mapping."""
    return {
        "plan_path": "plan.json",
        "n_reps": 2,
        "seed": 42,
        "generating": [
            {
                "name": "QRL",
                "model": "QRL",
                "sampling": {
                    "mode": "fixed",
                    "fixed": {"alpha": 0.2, "beta": 3.0},
                },
            }
        ],
        "candidates": [
            {
                "name": "QRL",
                "model": "QRL",
                "estimator": "comp_model_impl.estimators.mle_event_log.TransformedMLESubjectwiseEstimator",
                "estimator_kwargs": {"n_starts": 2, "maxiter": 20},
            }
        ],
    }


def test_coerce_mapping_and_float_mapping() -> None:
    """Mapping coercers should normalize keys and values."""
    assert _coerce_mapping(None, field_name="x") == {}
    assert _coerce_mapping({1: "a"}, field_name="x") == {"1": "a"}
    assert _coerce_float_mapping({"a": "1.5"}, field_name="x") == {"a": 1.5}

    with pytest.raises(TypeError):
        _ = _coerce_mapping(1, field_name="x")

    with pytest.raises(TypeError):
        _ = _coerce_float_mapping(1, field_name="x")


def test_coerce_component_ref_accepts_non_empty_string() -> None:
    """Component reference parser should accept non-empty strings."""
    assert _coerce_component_ref("QRL", field_name="model") == "QRL"
    assert _coerce_component_ref("  QRL  ", field_name="model") == "QRL"


def test_coerce_component_ref_rejects_invalid_input() -> None:
    """Component reference parser should reject malformed values."""
    with pytest.raises(ValueError):
        _ = _coerce_component_ref(None, field_name="model")

    with pytest.raises(ValueError):
        _ = _coerce_component_ref("   ", field_name="model")

    with pytest.raises(TypeError):
        _ = _coerce_component_ref({"factory": "QRL"}, field_name="model")

    with pytest.raises(TypeError):
        _ = _coerce_component_ref(123, field_name="model")


def test_coerce_sampling_spec_parses_by_condition() -> None:
    """Sampling parser should parse per-condition fields."""
    spec = _coerce_sampling_spec(
        {
            "mode": "fixed",
            "space": "param",
            "fixed": {"alpha": 0.2, "beta": 3.0},
            "by_condition": {
                "A": {"fixed": {"alpha": 0.1, "beta": 2.0}},
                "B": {"fixed": {"alpha": 0.3, "beta": 4.0}},
            },
        }
    )
    assert spec.mode == "fixed"
    assert spec.space == "param"
    assert spec.fixed["alpha"] == pytest.approx(0.2)
    assert set(spec.by_condition.keys()) == {"A", "B"}
    assert spec.by_condition["A"].fixed["beta"] == pytest.approx(2.0)


def test_config_from_raw_dict_new_schema() -> None:
    """Raw config should parse correctly using new schema fields."""
    cfg = config_from_raw_dict(_minimal_raw_config())

    assert isinstance(cfg, ModelRecoveryConfig)
    assert cfg.plan_path == "plan.json"
    assert cfg.n_reps == 2
    assert cfg.seed == 42
    assert isinstance(cfg.generating[0], GeneratingModelSpec)
    assert cfg.generating[0].model == "QRL"
    assert cfg.generating[0].model_kwargs == {}
    assert isinstance(cfg.candidates[0], CandidateModelSpec)
    assert cfg.candidates[0].estimator.endswith("TransformedMLESubjectwiseEstimator")
    assert cfg.candidates[0].estimator_kwargs["n_starts"] == 2


def test_config_from_raw_dict_rejects_legacy_factory_schema() -> None:
    """Legacy factory/kwargs format should be rejected."""
    raw = {
        "plan_path": "plan.json",
        "generating": [
            {
                "name": "QRL",
                "model": {"factory": "QRL", "kwargs": {"beta": 4.0}},
                "model_kwargs": {"beta": 5.0, "alpha": 0.2},
                "sampling": {"mode": "fixed", "fixed": {"alpha": 0.2, "beta": 5.0}},
            }
        ],
        "candidates": [
            {
                "name": "QRL",
                "model": {"factory": "QRL", "kwargs": {"beta": 3.0}},
                "model_kwargs": {"beta": 6.0},
                "estimator": {
                    "factory": "comp_model_impl.estimators.mle_event_log.TransformedMLESubjectwiseEstimator",
                    "kwargs": {"n_starts": 1, "maxiter": 10},
                },
                "estimator_kwargs": {"maxiter": 20},
                "fixed_params": {"alpha": 0.2},
            }
        ],
    }
    with pytest.raises(TypeError, match="must be a string"):
        _ = config_from_raw_dict(raw)


def test_config_from_raw_dict_generating_requires_model() -> None:
    """Generating entries should require explicit model references."""
    raw = {
        "plan_path": "plan.json",
        "generating": [
            {
                "name": "QRL",
                "sampling": {"mode": "fixed", "fixed": {"alpha": 0.2, "beta": 3.0}},
            }
        ],
        "candidates": [
            {
                "name": "QRL",
                "model": "QRL",
                "estimator": "comp_model_impl.estimators.mle_event_log.TransformedMLESubjectwiseEstimator",
            }
        ],
    }
    with pytest.raises(ValueError, match="generating\\[0\\] missing required 'model'"):
        _ = config_from_raw_dict(raw)


def test_config_from_raw_dict_rejects_missing_required_fields() -> None:
    """Config parser should reject malformed top-level entries."""
    with pytest.raises(ValueError, match="plan_path"):
        _ = config_from_raw_dict({})

    raw = _minimal_raw_config()
    del raw["candidates"][0]["name"]
    with pytest.raises(ValueError, match="name"):
        _ = config_from_raw_dict(raw)

    raw = _minimal_raw_config()
    del raw["candidates"][0]["model"]
    with pytest.raises(ValueError, match="model"):
        _ = config_from_raw_dict(raw)

    raw = _minimal_raw_config()
    del raw["candidates"][0]["estimator"]
    with pytest.raises(ValueError, match="estimator"):
        _ = config_from_raw_dict(raw)

    raw = _minimal_raw_config()
    del raw["generating"][0]["model"]
    with pytest.raises(ValueError, match="generating\\[0\\] missing required 'model'"):
        _ = config_from_raw_dict(raw)


def test_load_model_recovery_config_json_roundtrip(tmp_path: Path) -> None:
    """JSON config loader should parse into ModelRecoveryConfig."""
    raw = _minimal_raw_config()
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(raw), encoding="utf-8")

    cfg = load_model_recovery_config(p)
    assert isinstance(cfg, ModelRecoveryConfig)
    assert cfg.plan_path == "plan.json"
    assert cfg.selection.criterion == "bic"


def test_load_model_recovery_config_yaml_roundtrip(tmp_path: Path) -> None:
    """YAML config loader should parse into ModelRecoveryConfig."""
    yaml = pytest.importorskip("yaml")
    raw = _minimal_raw_config()
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")

    cfg = load_model_recovery_config(p)
    assert isinstance(cfg, ModelRecoveryConfig)
    assert cfg.candidates[0].name == "QRL"


def test_load_model_recovery_config_errors(tmp_path: Path) -> None:
    """Config loader should fail for bad paths, suffixes, and root types."""
    with pytest.raises(FileNotFoundError):
        _ = load_model_recovery_config(tmp_path / "missing.json")

    p_txt = tmp_path / "cfg.txt"
    p_txt.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported config extension"):
        _ = load_model_recovery_config(p_txt)

    p_json = tmp_path / "cfg.json"
    p_json.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError, match="mapping"):
        _ = load_model_recovery_config(p_json)


def test_config_to_json_contains_new_fields() -> None:
    """Serialized config should include model and estimator kwargs fields."""
    cfg = config_from_raw_dict(_minimal_raw_config())
    dumped = config_to_json(cfg)
    parsed = json.loads(dumped)

    cand = parsed["candidates"][0]
    assert cand["model"] == "QRL"
    assert "model_kwargs" in cand
    assert "estimator_kwargs" in cand
