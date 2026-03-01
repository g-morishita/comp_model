"""Tests for hierarchical fitting serialization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from comp_model.core.contracts import DecisionContext
from comp_model.core.data import BlockData, StudyData, SubjectData
from comp_model.inference import (
    fit_study_hierarchical_map,
    hierarchical_study_block_records,
    hierarchical_study_summary_records,
    write_hierarchical_study_block_records_csv,
    write_hierarchical_study_summary_csv,
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


def _make_block(block_id: str, p_right: float, seed: int) -> BlockData:
    """Generate one synthetic block."""

    trace = run_episode(
        problem=StationaryBanditProblem([0.5, 0.5]),
        model=FixedChoiceModel(p_right=p_right),
        config=SimulationConfig(n_trials=20, seed=seed),
    )
    return BlockData(block_id=block_id, event_trace=trace)


def _make_study_result():
    """Fit a minimal study and return hierarchical MAP result."""

    study = StudyData(
        subjects=(
            SubjectData(
                subject_id="s1",
                blocks=(
                    _make_block("b1", 0.2, 1),
                    _make_block("b2", 0.7, 2),
                ),
            ),
            SubjectData(
                subject_id="s2",
                blocks=(
                    _make_block("b1", 0.3, 3),
                    _make_block("b2", 0.8, 4),
                ),
            ),
        )
    )
    return fit_study_hierarchical_map(
        study,
        model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
        parameter_names=("p_right",),
        transforms={"p_right": unit_interval_logit_transform()},
        initial_group_location={"p_right": 0.5},
        initial_group_scale={"p_right": 0.5},
    )


def test_hierarchical_record_helpers_return_rows() -> None:
    """Record helpers should flatten hierarchical results into row dictionaries."""

    result = _make_study_result()
    block_rows = hierarchical_study_block_records(result)
    summary_rows = hierarchical_study_summary_records(result)

    assert len(block_rows) == 4
    assert len(summary_rows) == 2
    assert set(block_rows[0]) >= {"subject_id", "block_id", "log_likelihood", "param__p_right"}
    assert set(summary_rows[0]) >= {"subject_id", "total_log_likelihood", "total_log_prior", "total_log_posterior"}


def test_write_hierarchical_csv_helpers(tmp_path) -> None:
    """CSV writer helpers should persist hierarchical rows to disk."""

    result = _make_study_result()
    block_path = tmp_path / "hierarchical_blocks.csv"
    summary_path = tmp_path / "hierarchical_summary.csv"

    out_block = write_hierarchical_study_block_records_csv(result, block_path)
    out_summary = write_hierarchical_study_summary_csv(result, summary_path)

    assert out_block.exists()
    assert out_summary.exists()
    assert out_block.read_text(encoding="utf-8").startswith("subject_id,")
    assert out_summary.read_text(encoding="utf-8").startswith("subject_id,")
