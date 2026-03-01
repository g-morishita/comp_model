"""Tests for Stan-backed hierarchical posterior sampling helpers."""

from __future__ import annotations

import numpy as np
import pytest

import comp_model.inference.hierarchical_stan as hierarchical_stan_module
from comp_model.core.data import BlockData, SubjectData, TrialDecision
from comp_model.inference.hierarchical_stan import (
    sample_subject_hierarchical_posterior_stan,
)


def _trial(trial_index: int, action: int, reward: float) -> TrialDecision:
    """Build one trial row for hierarchical Stan tests."""

    return TrialDecision(
        trial_index=trial_index,
        decision_index=0,
        actor_id="subject",
        available_actions=(0, 1),
        action=action,
        observation={"state": 0},
        outcome={"reward": reward},
    )


class _FakeFit:
    """Small fake CmdStan fit object for unit testing."""

    def __init__(self, variables: dict[str, np.ndarray]) -> None:
        self._variables = variables

    def stan_variable(self, name: str) -> np.ndarray:
        """Return pre-seeded variable draws."""

        return self._variables[name]


def test_sample_subject_hierarchical_posterior_stan_decodes_draws(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stan helper should decode backend draws into public result dataclasses."""

    block_1 = BlockData(
        block_id="b1",
        trials=(_trial(0, 1, 1.0), _trial(1, 0, 0.0), _trial(2, 1, 1.0)),
    )
    block_2 = BlockData(
        block_id="b2",
        trials=(_trial(0, 0, 0.0), _trial(1, 1, 1.0), _trial(2, 1, 1.0)),
    )
    subject = SubjectData(subject_id="s1", blocks=(block_1, block_2))

    n_draws = 8
    fake_fit = _FakeFit(
        variables={
            "group_loc_z": np.linspace(-0.1, 0.2, n_draws).reshape(n_draws, 1),
            "group_log_scale": np.linspace(-1.0, -0.7, n_draws).reshape(n_draws, 1),
            "block_z": np.full((n_draws, 2, 1), 0.1, dtype=float),
            "block_param": np.full((n_draws, 2, 1), 0.52, dtype=float),
            "log_likelihood_total": np.linspace(-5.0, -4.0, n_draws),
            "log_prior_total": np.linspace(-1.0, -0.5, n_draws),
            "log_posterior_total": np.linspace(-6.0, -4.5, n_draws),
        }
    )

    monkeypatch.setattr(
        hierarchical_stan_module,
        "_run_stan_hierarchical_nuts",
        lambda **kwargs: fake_fit,
    )

    result = sample_subject_hierarchical_posterior_stan(
        subject,
        model_component_id="asocial_state_q_value_softmax",
        model_kwargs={"beta": 2.0, "initial_value": 0.0},
        parameter_names=["alpha"],
        transform_kinds={"alpha": "unit_interval_logit"},
        n_samples=4,
        n_warmup=3,
        thin=1,
        n_chains=2,
        random_seed=10,
    )

    assert result.subject_id == "s1"
    assert result.parameter_names == ("alpha",)
    assert len(result.draws) == n_draws
    assert result.draws[0].candidate.block_params[0]["alpha"] == pytest.approx(0.52)
    assert result.draws[0].candidate.log_likelihood == pytest.approx(-5.0)
    assert result.diagnostics.method == "within_subject_hierarchical_stan_nuts"
    assert result.diagnostics.n_kept_draws == n_draws


def test_sample_subject_hierarchical_posterior_stan_rejects_unsupported_model() -> None:
    """Stan helper should fail fast on unsupported model component IDs."""

    block = BlockData(
        block_id="b1",
        trials=(_trial(0, 1, 1.0), _trial(1, 0, 0.0)),
    )
    subject = SubjectData(subject_id="s1", blocks=(block,))

    with pytest.raises(ValueError, match="currently supports only"):
        sample_subject_hierarchical_posterior_stan(
            subject,
            model_component_id="social_observed_outcome_q",
            model_kwargs={},
            parameter_names=["alpha"],
        )

