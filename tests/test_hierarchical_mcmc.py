"""Tests for within-subject hierarchical posterior sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pytest

from comp_model.core.contracts import DecisionContext
from comp_model.core.data import BlockData, StudyData, SubjectData
from comp_model.inference import (
    sample_study_hierarchical_posterior,
    sample_subject_hierarchical_posterior,
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


def test_sample_subject_hierarchical_posterior_runs_and_returns_draws() -> None:
    """Hierarchical MCMC should retain requested number of draws."""

    subject = _make_subject()
    result = sample_subject_hierarchical_posterior(
        subject,
        model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
        parameter_names=("p_right",),
        transforms={"p_right": unit_interval_logit_transform()},
        initial_group_location={"p_right": 0.5},
        initial_group_scale={"p_right": 0.5},
        n_samples=30,
        n_warmup=20,
        thin=2,
        random_seed=123,
    )

    assert result.subject_id == "s1"
    assert result.parameter_names == ("p_right",)
    assert len(result.draws) == 30
    assert result.diagnostics.n_kept_draws == 30
    assert result.diagnostics.n_warmup == 20
    assert result.n_blocks == 3
    assert math.isfinite(result.map_candidate.log_likelihood)
    assert math.isfinite(result.map_candidate.log_prior)
    assert math.isfinite(result.map_candidate.log_posterior)

    for draw in result.draws:
        assert len(draw.candidate.block_params) == 3
        assert 0.0 < draw.candidate.group_scale_z["p_right"]
        for block_params in draw.candidate.block_params:
            assert 0.0 < block_params["p_right"] < 1.0


def test_sample_subject_hierarchical_posterior_is_seed_deterministic() -> None:
    """Equal random seeds should produce identical retained draw trajectories."""

    subject = _make_subject()
    kwargs = dict(
        model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
        parameter_names=("p_right",),
        transforms={"p_right": unit_interval_logit_transform()},
        initial_group_location={"p_right": 0.5},
        initial_group_scale={"p_right": 0.5},
        n_samples=10,
        n_warmup=5,
        thin=1,
        random_seed=77,
    )
    result_a = sample_subject_hierarchical_posterior(subject, **kwargs)
    result_b = sample_subject_hierarchical_posterior(subject, **kwargs)

    draws_a = [draw.candidate.log_posterior for draw in result_a.draws]
    draws_b = [draw.candidate.log_posterior for draw in result_b.draws]
    assert draws_a == draws_b


def test_sample_subject_hierarchical_posterior_validates_initial_block_length() -> None:
    """Initial block parameter list must align with subject block count."""

    subject = _make_subject()
    with pytest.raises(ValueError, match="must match number of subject blocks"):
        sample_subject_hierarchical_posterior(
            subject,
            model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
            parameter_names=("p_right",),
            transforms={"p_right": unit_interval_logit_transform()},
            initial_block_params=({"p_right": 0.5},),
            n_samples=5,
            n_warmup=2,
        )


def test_sample_study_hierarchical_posterior_runs_all_subjects() -> None:
    """Study-level wrapper should sample all subjects independently."""

    study = StudyData(
        subjects=(
            _make_subject(),
            SubjectData(
                subject_id="s2",
                blocks=(
                    _make_block(block_id="b1", p_right=0.3, seed=11),
                    _make_block(block_id="b2", p_right=0.6, seed=12),
                ),
            ),
        )
    )
    result = sample_study_hierarchical_posterior(
        study,
        model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
        parameter_names=("p_right",),
        transforms={"p_right": unit_interval_logit_transform()},
        initial_group_location={"p_right": 0.5},
        initial_group_scale={"p_right": 0.5},
        n_samples=12,
        n_warmup=6,
        thin=2,
        random_seed=999,
    )

    assert result.n_subjects == 2
    assert len(result.subject_results) == 2
    assert math.isfinite(result.total_map_log_likelihood)
    assert math.isfinite(result.total_map_log_prior)
    assert math.isfinite(result.total_map_log_posterior)
