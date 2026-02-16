"""Tests for helper utilities in parameter recovery run module."""

from __future__ import annotations

from pathlib import Path

import pytest

from comp_model_core.plans.block import BlockPlan, StudyPlan
from comp_model_impl.recovery.parameter.run import (
    _plan_summary,
    _strip_hat_key,
    make_unique_run_dir,
)


def test_strip_hat_key() -> None:
    """_strip_hat_key should remove a trailing _hat suffix."""
    assert _strip_hat_key("alpha_hat") == "alpha"
    assert _strip_hat_key("beta") == "beta"


def test_make_unique_run_dir_creates_dir(tmp_path: Path) -> None:
    """make_unique_run_dir should create a unique directory."""
    out_dir = make_unique_run_dir(tmp_path)
    assert out_dir.exists()
    assert out_dir.is_dir()


def test_plan_summary_counts_blocks_and_trials() -> None:
    """_plan_summary should summarize blocks and trials per subject."""
    plan = StudyPlan(
        subjects={
            "s1": [
                BlockPlan(
                    block_id="b1",
                    n_trials=2,
                    condition="c1",
                    bandit_type="BernoulliBanditEnv",
                    bandit_config={"probs": [0.2, 0.8]},
                    trial_specs=[{"self_outcome": {"kind": "VERIDICAL"}, "available_actions": [0, 1]}] * 2,
                )
            ],
            "s2": [
                BlockPlan(
                    block_id="b1",
                    n_trials=3,
                    condition="c1",
                    bandit_type="BernoulliBanditEnv",
                    bandit_config={"probs": [0.2, 0.8]},
                    trial_specs=[{"self_outcome": {"kind": "VERIDICAL"}, "available_actions": [0, 1]}] * 3,
                )
            ],
        }
    )

    summary = _plan_summary(plan)
    assert summary["n_subjects"] == 2
    assert summary["n_blocks_unique"] == 1
    assert summary["blocks_per_subject_min"] == 1
    assert summary["total_trials_per_subject_min"] == 2
    assert summary["total_trials_per_subject_max"] == 3
    blocks_table = summary["blocks_table"]
    assert blocks_table[0]["block_id"] == "b1"
    assert blocks_table[0]["n_trials_min"] == 2
    assert blocks_table[0]["n_trials_max"] == 3
