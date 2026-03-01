"""Tests for hierarchical MCMC serialization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from comp_model.core.contracts import DecisionContext
from comp_model.core.data import BlockData, StudyData, SubjectData
from comp_model.inference import (
    hierarchical_mcmc_study_draw_records,
    hierarchical_mcmc_study_summary_records,
    sample_study_hierarchical_posterior,
    write_hierarchical_mcmc_study_draw_records_csv,
    write_hierarchical_mcmc_study_summary_csv,
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
    """Sample a minimal study and return hierarchical posterior result."""

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
    return sample_study_hierarchical_posterior(
        study,
        model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
        parameter_names=("p_right",),
        transforms={"p_right": unit_interval_logit_transform()},
        initial_group_location={"p_right": 0.5},
        initial_group_scale={"p_right": 0.5},
        n_samples=10,
        n_warmup=8,
        thin=1,
        random_seed=101,
    )


def test_hierarchical_mcmc_record_helpers_return_rows() -> None:
    """Record helpers should flatten hierarchical MCMC results into rows."""

    result = _make_study_result()
    draw_rows = hierarchical_mcmc_study_draw_records(result)
    summary_rows = hierarchical_mcmc_study_summary_records(result)

    assert len(draw_rows) == 20
    assert len(summary_rows) == 2
    assert set(draw_rows[0]) >= {
        "subject_id",
        "iteration",
        "accepted",
        "log_likelihood",
        "log_prior",
        "log_posterior",
        "group_location_z__p_right",
        "group_scale__p_right",
        "block_0__param__p_right",
    }
    assert set(summary_rows[0]) >= {
        "subject_id",
        "n_draws",
        "n_blocks",
        "acceptance_rate",
        "map_log_likelihood",
        "map_log_prior",
        "map_log_posterior",
    }


def test_write_hierarchical_mcmc_csv_helpers(tmp_path) -> None:
    """CSV writer helpers should persist hierarchical MCMC rows to disk."""

    result = _make_study_result()
    draw_path = tmp_path / "hierarchical_mcmc_draws.csv"
    summary_path = tmp_path / "hierarchical_mcmc_summary.csv"

    out_draw = write_hierarchical_mcmc_study_draw_records_csv(result, draw_path)
    out_summary = write_hierarchical_mcmc_study_summary_csv(result, summary_path)

    assert out_draw.exists()
    assert out_summary.exists()
    assert out_draw.read_text(encoding="utf-8").startswith("subject_id,")
    assert out_summary.read_text(encoding="utf-8").startswith("subject_id,")
