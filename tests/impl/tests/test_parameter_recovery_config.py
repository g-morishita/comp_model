"""Tests for parameter recovery config loading."""

from __future__ import annotations

import json
from pathlib import Path

from comp_model_impl.recovery.parameter.config import load_parameter_recovery_config


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_load_parameter_recovery_config_n_jobs_default(tmp_path: Path) -> None:
    """n_jobs should default to 1 when omitted."""
    cfg_path = tmp_path / "recovery.json"
    _write_json(
        cfg_path,
        {
            "plan_path": "dummy_plan.json",
            "n_reps": 2,
            "seed": 7,
            "sampling": {"mode": "fixed", "fixed": {"alpha": 0.2}},
        },
    )
    cfg = load_parameter_recovery_config(cfg_path)
    assert cfg.n_jobs == 1


def test_load_parameter_recovery_config_n_jobs_clamped(tmp_path: Path) -> None:
    """n_jobs should be clamped to at least 1."""
    cfg_path = tmp_path / "recovery.json"
    _write_json(
        cfg_path,
        {
            "plan_path": "dummy_plan.json",
            "n_reps": 2,
            "seed": 7,
            "n_jobs": 0,
            "sampling": {"mode": "fixed", "fixed": {"alpha": 0.2}},
        },
    )
    cfg = load_parameter_recovery_config(cfg_path)
    assert cfg.n_jobs == 1

