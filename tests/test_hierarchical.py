"""Tests for within-subject hierarchical MAP fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math
import pytest

from comp_model.core.contracts import DecisionContext
from comp_model.core.data import BlockData, SubjectData
from comp_model.inference import fit_subject_hierarchical_map
from comp_model.inference.transforms import unit_interval_logit_transform
from comp_model.problems import StationaryBanditProblem
from comp_model.runtime import SimulationConfig, run_episode


@dataclass
class FixedChoiceModel:
    """Toy model with one free right-choice probability parameter."""

    p_right: float

    def start_episode(self) -> None:
        """No-op reset."""

    def action_distribution(
        self,
        observation: Any,
        *,
        context: DecisionContext[int],
    ) -> dict[int, float]:
        """Return fixed Bernoulli action probabilities."""

        assert context.available_actions == (0, 1)
        return {0: 1.0 - self.p_right, 1: self.p_right}

    def update(
        self,
        observation: Any,
        action: int,
        outcome: Any,
        *,
        context: DecisionContext[int],
    ) -> None:
        """No-op update."""


def _make_block(*, block_id: str, p_right: float, seed: int, n_trials: int = 40) -> BlockData:
    """Generate one synthetic block trace."""

    trace = run_episode(
        problem=StationaryBanditProblem([0.5, 0.5]),
        model=FixedChoiceModel(p_right=p_right),
        config=SimulationConfig(n_trials=n_trials, seed=seed),
    )
    return BlockData(block_id=block_id, event_trace=trace)


def _make_subject() -> SubjectData:
    """Build one multi-block synthetic subject for hierarchical tests."""

    return SubjectData(
        subject_id="s1",
        blocks=(
            _make_block(block_id="b1", p_right=0.2, seed=1),
            _make_block(block_id="b2", p_right=0.5, seed=2),
            _make_block(block_id="b3", p_right=0.8, seed=3),
        ),
    )


def test_fit_subject_hierarchical_map_runs_and_returns_block_parameters() -> None:
    """Hierarchical MAP should fit all subject blocks with pooled parameters."""

    subject = _make_subject()
    result = fit_subject_hierarchical_map(
        subject,
        model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
        parameter_names=("p_right",),
        transforms={"p_right": unit_interval_logit_transform()},
        initial_group_location={"p_right": 0.5},
        initial_group_scale={"p_right": 0.5},
    )

    assert result.subject_id == "s1"
    assert result.parameter_names == ("p_right",)
    assert len(result.block_results) == 3
    assert result.group_scale_z["p_right"] > 0.0
    assert math.isfinite(result.total_log_likelihood)
    assert math.isfinite(result.total_log_prior)
    assert math.isfinite(result.total_log_posterior)

    for block in result.block_results:
        assert 0.0 < block.params["p_right"] < 1.0
        assert math.isfinite(block.log_likelihood)


def test_fit_subject_hierarchical_map_validates_initial_block_length() -> None:
    """Initial block parameter list must align with subject block count."""

    subject = _make_subject()
    with pytest.raises(ValueError, match="must match number of subject blocks"):
        fit_subject_hierarchical_map(
            subject,
            model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
            parameter_names=("p_right",),
            transforms={"p_right": unit_interval_logit_transform()},
            initial_block_params=({"p_right": 0.5},),
        )


def test_fit_subject_hierarchical_map_validates_positive_group_scale() -> None:
    """Group-scale initialization must be positive."""

    subject = _make_subject()
    with pytest.raises(ValueError, match="must be > 0"):
        fit_subject_hierarchical_map(
            subject,
            model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
            parameter_names=("p_right",),
            transforms={"p_right": unit_interval_logit_transform()},
            initial_group_scale={"p_right": 0.0},
        )
