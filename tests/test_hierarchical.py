"""Tests for removed SciPy hierarchical MAP APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from comp_model.core.contracts import DecisionContext
from comp_model.core.data import BlockData, StudyData, SubjectData
from comp_model.inference.hierarchical import (
    fit_study_hierarchical_map,
    fit_subject_hierarchical_map,
)
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


def _make_block(*, block_id: str, p_right: float, seed: int, n_trials: int = 20) -> BlockData:
    """Generate one synthetic block trace."""

    trace = run_episode(
        problem=StationaryBanditProblem([0.5, 0.5]),
        model=FixedChoiceModel(p_right=p_right),
        config=SimulationConfig(n_trials=n_trials, seed=seed),
    )
    return BlockData(block_id=block_id, event_trace=trace)


def test_fit_subject_hierarchical_map_is_removed() -> None:
    """Subject-level SciPy hierarchical MAP should be removed."""

    subject = SubjectData(
        subject_id="s1",
        blocks=(
            _make_block(block_id="b1", p_right=0.2, seed=1),
            _make_block(block_id="b2", p_right=0.8, seed=2),
        ),
    )
    with pytest.raises(RuntimeError, match="has been removed"):
        fit_subject_hierarchical_map(
            subject,
            model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
            parameter_names=("p_right",),
            transforms={"p_right": unit_interval_logit_transform()},
        )


def test_fit_study_hierarchical_map_is_removed() -> None:
    """Study-level SciPy hierarchical MAP should be removed."""

    subject = SubjectData(
        subject_id="s1",
        blocks=(_make_block(block_id="b1", p_right=0.2, seed=1),),
    )
    study = StudyData(subjects=(subject,))
    with pytest.raises(RuntimeError, match="has been removed"):
        fit_study_hierarchical_map(
            study,
            model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
            parameter_names=("p_right",),
            transforms={"p_right": unit_interval_logit_transform()},
        )
