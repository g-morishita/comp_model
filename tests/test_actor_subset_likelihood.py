"""Tests for actor-subset replay likelihood on multi-actor traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from comp_model.core.contracts import DecisionContext
from comp_model.demonstrators import FixedSequenceDemonstrator
from comp_model.inference import (
    ActionReplayLikelihood,
    ActorSubsetReplayLikelihood,
    FitSpec,
    fit_model,
)
from comp_model.models import UniformRandomPolicyModel
from comp_model.problems import TwoStageSocialBanditProgram
from comp_model.runtime import SimulationConfig, run_trial_program


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

        del observation
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

        del observation, action, outcome, context


def _social_trace(*, n_trials: int, seed: int):
    """Generate one two-actor social trace for likelihood tests."""

    return run_trial_program(
        program=TwoStageSocialBanditProgram([0.5, 0.5]),
        models={
            "subject": UniformRandomPolicyModel(),
            "demonstrator": FixedSequenceDemonstrator(sequence=[1] * n_trials),
        },
        config=SimulationConfig(n_trials=n_trials, seed=seed),
    )


def test_action_replay_likelihood_rejects_unmodeled_actor_on_social_trace() -> None:
    """Single-actor replay likelihood should fail on multi-actor traces."""

    trace = _social_trace(n_trials=10, seed=1)
    with pytest.raises(ValueError, match="unknown actor_id"):
        ActionReplayLikelihood().evaluate(trace, UniformRandomPolicyModel())


def test_actor_subset_likelihood_scores_subject_only_with_trace_autofill() -> None:
    """Actor-subset likelihood should replay all actors and score selected ones."""

    trace = _social_trace(n_trials=12, seed=2)
    result = ActorSubsetReplayLikelihood(
        fitted_actor_id="subject",
        scored_actor_ids=("subject",),
        auto_fill_unmodeled_actors=True,
    ).evaluate(trace, UniformRandomPolicyModel())

    assert len(result.steps) == 12
    assert all(step.actor_id == "subject" for step in result.steps)
    assert np.isfinite(result.total_log_likelihood)


def test_actor_subset_likelihood_without_autofill_requires_all_actor_models() -> None:
    """Actor-subset likelihood should fail if trace actors are not modeled."""

    trace = _social_trace(n_trials=6, seed=3)
    with pytest.raises(ValueError, match="unknown actor_id"):
        ActorSubsetReplayLikelihood(
            fitted_actor_id="subject",
            scored_actor_ids=("subject",),
            auto_fill_unmodeled_actors=False,
        ).evaluate(trace, UniformRandomPolicyModel())


def test_fit_model_supports_social_trace_with_actor_subset_likelihood() -> None:
    """fit_model should recover subject policy on social traces via actor subset likelihood."""

    n_trials = 80
    trace = run_trial_program(
        program=TwoStageSocialBanditProgram([0.5, 0.5]),
        models={
            "subject": FixedChoiceModel(p_right=0.8),
            "demonstrator": FixedSequenceDemonstrator(sequence=[1] * n_trials),
        },
        config=SimulationConfig(n_trials=n_trials, seed=9),
    )

    fit = fit_model(
        trace,
        model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
        fit_spec=FitSpec(
            estimator_type="grid_search",
            parameter_grid={"p_right": [0.2, 0.5, 0.8]},
        ),
        likelihood_program=ActorSubsetReplayLikelihood(
            fitted_actor_id="subject",
            scored_actor_ids=("subject",),
            auto_fill_unmodeled_actors=True,
        ),
    )
    assert fit.best.params["p_right"] == pytest.approx(0.8)
