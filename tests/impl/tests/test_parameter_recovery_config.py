"""Tests for parameter recovery config loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from comp_model_impl.recovery.parameter.config import load_parameter_recovery_config


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj), encoding="utf-8")


def _valid_cfg_dict() -> dict:
    return {
        "plan_path": "dummy_plan.json",
        "n_reps": 2,
        "seed": 7,
        "n_jobs": 1,
        "components": {
            "generator": {"name": "EventLogAsocialGenerator", "kwargs": {}},
            "generating_model": {"name": "QRL", "kwargs": {}},
            "fitting_model": {"name": "QRL", "kwargs": {}},
            "estimator": {"name": "BoxMLESubjectwiseEstimator", "kwargs": {"n_starts": 5}},
        },
        "sampling": {
            "mode": "fixed",
            "space": "param",
            "fixed": {"alpha": 0.2},
            "by_condition": {},
            "clip_to_bounds": True,
        },
        "output": {
            "out_dir": "recovery_out",
            "save_format": "csv",
            "save_config": True,
            "save_fit_diagnostics": True,
            "save_simulated_study": False,
        },
    }


def test_load_parameter_recovery_config_requires_all_top_level_fields(tmp_path: Path) -> None:
    """Missing required top-level fields should raise ValueError."""
    for missing_key in ("plan_path", "n_reps", "seed", "n_jobs", "components", "sampling"):
        cfg = _valid_cfg_dict()
        cfg.pop(missing_key)

        cfg_path = tmp_path / f"missing_{missing_key}.json"
        _write_json(cfg_path, cfg)

        with pytest.raises(ValueError, match=rf"Missing required field: {missing_key}"):
            _ = load_parameter_recovery_config(cfg_path)


def test_load_parameter_recovery_config_requires_component_fields(tmp_path: Path) -> None:
    """Components section should require all component entries."""
    for missing_key in ("generator", "generating_model", "fitting_model", "estimator"):
        cfg = _valid_cfg_dict()
        cfg["components"].pop(missing_key)

        cfg_path = tmp_path / f"missing_components_{missing_key}.json"
        _write_json(cfg_path, cfg)

        with pytest.raises(ValueError, match=rf"Missing required field: {missing_key}"):
            _ = load_parameter_recovery_config(cfg_path)


def test_load_parameter_recovery_config_rejects_invalid_component_name(tmp_path: Path) -> None:
    """Component names should be non-empty strings."""
    cfg = _valid_cfg_dict()
    cfg["components"]["generator"]["name"] = ""

    cfg_path = tmp_path / "bad_component_name.json"
    _write_json(cfg_path, cfg)

    with pytest.raises(ValueError, match=r"components\.generator\.name must be a non-empty string"):
        _ = load_parameter_recovery_config(cfg_path)


def test_load_parameter_recovery_config_rejects_unknown_component_names(tmp_path: Path) -> None:
    """Component names should exist in the default implementation registry."""
    cases = [
        ("generator", r"Unknown components\.generator\.name"),
        ("generating_model", r"Unknown components\.generating_model\.name"),
        ("fitting_model", r"Unknown components\.fitting_model\.name"),
        ("estimator", r"Unknown components\.estimator\.name"),
    ]

    for key, msg in cases:
        cfg = _valid_cfg_dict()
        cfg["components"][key]["name"] = "NotRegistered"

        cfg_path = tmp_path / f"unknown_components_{key}.json"
        _write_json(cfg_path, cfg)

        with pytest.raises(ValueError, match=msg):
            _ = load_parameter_recovery_config(cfg_path)


def test_load_parameter_recovery_config_requires_common_sampling_fields(tmp_path: Path) -> None:
    """Sampling section must include strict no-default common fields."""
    for missing_key in ("mode", "space", "by_condition", "clip_to_bounds"):
        cfg = _valid_cfg_dict()
        cfg["sampling"].pop(missing_key)

        cfg_path = tmp_path / f"missing_sampling_{missing_key}.json"
        _write_json(cfg_path, cfg)

        with pytest.raises(ValueError, match=rf"Missing required field: {missing_key}"):
            _ = load_parameter_recovery_config(cfg_path)


def test_load_parameter_recovery_config_requires_mode_specific_sampling_fields(tmp_path: Path) -> None:
    """Mode-specific sampling fields should be required."""
    cases = [
        ("fixed", "fixed"),
        ("independent", "individual"),
        ("hierarchical", "population"),
        ("hierarchical", "individual_sd"),
    ]

    for mode, missing_key in cases:
        cfg = _valid_cfg_dict()
        cfg["sampling"]["mode"] = mode

        if mode == "independent":
            cfg["sampling"].pop("fixed", None)
            cfg["sampling"]["individual"] = {"alpha": {"name": "constant", "args": {"value": 0.2}}}
        elif mode == "hierarchical":
            cfg["sampling"].pop("fixed", None)
            cfg["sampling"]["population"] = {"alpha": {"name": "constant", "args": {"value": 0.2}}}
            cfg["sampling"]["individual_sd"] = {"alpha": 0.05}

        cfg["sampling"].pop(missing_key, None)

        cfg_path = tmp_path / f"missing_{mode}_{missing_key}.json"
        _write_json(cfg_path, cfg)

        with pytest.raises(ValueError, match=rf"Missing required field: {missing_key}"):
            _ = load_parameter_recovery_config(cfg_path)


def test_load_parameter_recovery_config_rejects_nonpositive_n_reps(tmp_path: Path) -> None:
    """n_reps must be >= 1."""
    cfg = _valid_cfg_dict()
    cfg["n_reps"] = 0
    cfg_path = tmp_path / "bad_n_reps_0.json"
    _write_json(cfg_path, cfg)

    with pytest.raises(ValueError, match=r"n_reps must be >= 1"):
        _ = load_parameter_recovery_config(cfg_path)


def test_load_parameter_recovery_config_rejects_non_int_n_reps(tmp_path: Path) -> None:
    """n_reps must be an int (bool is not allowed)."""
    cfg = _valid_cfg_dict()
    cfg["n_reps"] = True
    cfg_path = tmp_path / "bad_n_reps_bool.json"
    _write_json(cfg_path, cfg)

    with pytest.raises(ValueError, match=r"n_reps must be an integer"):
        _ = load_parameter_recovery_config(cfg_path)


def test_load_parameter_recovery_config_accepts_valid_strict_config(tmp_path: Path) -> None:
    """A fully specified config should load successfully."""
    cfg_path = tmp_path / "valid.json"
    _write_json(cfg_path, _valid_cfg_dict())

    cfg = load_parameter_recovery_config(cfg_path)
    assert cfg.plan_path == "dummy_plan.json"
    assert cfg.n_reps == 2
    assert cfg.seed == 7
    assert cfg.n_jobs == 1
    assert cfg.components.generator.name == "EventLogAsocialGenerator"
    assert cfg.components.generating_model.name == "QRL"
    assert cfg.components.fitting_model.name == "QRL"
    assert cfg.components.estimator.name == "BoxMLESubjectwiseEstimator"
    assert cfg.components.estimator.kwargs["n_starts"] == 5
    assert cfg.sampling.mode == "fixed"
    assert cfg.sampling.space == "param"
    assert cfg.sampling.fixed["alpha"] == pytest.approx(0.2)
