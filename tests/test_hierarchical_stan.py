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


def _social_trials() -> tuple[TrialDecision, ...]:
    """Build one two-trial social row sequence (demo then subject)."""

    return (
        TrialDecision(
            trial_index=0,
            decision_index=0,
            actor_id="demonstrator",
            learner_id="subject",
            available_actions=(0, 1),
            action=1,
            observation={"state": 0, "stage": "demonstrator"},
            outcome={"reward": 1.0, "source_actor_id": "demonstrator"},
        ),
        TrialDecision(
            trial_index=0,
            decision_index=1,
            actor_id="subject",
            learner_id="subject",
            available_actions=(0, 1),
            action=0,
            observation={"state": 0, "stage": "subject", "demonstrator_action": 1},
            outcome={"reward": 0.0, "source_actor_id": "subject"},
        ),
        TrialDecision(
            trial_index=1,
            decision_index=0,
            actor_id="demonstrator",
            learner_id="subject",
            available_actions=(0, 1),
            action=0,
            observation={"state": 0, "stage": "demonstrator"},
            outcome={"reward": 0.0, "source_actor_id": "demonstrator"},
        ),
        TrialDecision(
            trial_index=1,
            decision_index=1,
            actor_id="subject",
            learner_id="subject",
            available_actions=(0, 1),
            action=1,
            observation={"state": 0, "stage": "subject", "demonstrator_action": 0},
            outcome={"reward": 1.0, "source_actor_id": "subject"},
        ),
    )


class _FakeFit:
    """Small fake CmdStan fit object for unit testing."""

    def __init__(self, variables: dict[str, np.ndarray]) -> None:
        self._variables = variables

    def stan_variable(self, name: str) -> np.ndarray:
        """Return pre-seeded variable draws."""

        return self._variables[name]


def _fake_fit(*, n_draws: int, n_blocks: int, n_params: int, block_param_value: float = 0.52) -> _FakeFit:
    """Construct a fake CmdStan fit object with compatible variable shapes."""

    return _FakeFit(
        variables={
            "group_loc_z": np.linspace(-0.1, 0.2, n_draws * n_params).reshape(n_draws, n_params),
            "group_log_scale": np.linspace(-1.0, -0.7, n_draws * n_params).reshape(n_draws, n_params),
            "block_z": np.full((n_draws, n_blocks, n_params), 0.1, dtype=float),
            "block_param": np.full((n_draws, n_blocks, n_params), block_param_value, dtype=float),
            "log_likelihood_total": np.linspace(-5.0, -4.0, n_draws),
            "log_prior_total": np.linspace(-1.0, -0.5, n_draws),
            "log_posterior_total": np.linspace(-6.0, -4.5, n_draws),
        }
    )


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
    fake_fit = _fake_fit(n_draws=n_draws, n_blocks=2, n_params=1, block_param_value=0.52)

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


@pytest.mark.parametrize(
    ("component_id", "parameter_name"),
    [
        ("social_observed_outcome_q", "alpha_observed"),
        ("social_observed_outcome_q_perseveration", "alpha_observed"),
        ("social_observed_outcome_value_shaping", "alpha_social"),
        ("social_observed_outcome_value_shaping_perseveration", "alpha_social"),
        ("social_policy_reliability_gated_value_shaping", "alpha_social_base"),
        ("social_constant_demo_bias_observed_outcome_q_perseveration", "demo_bias"),
        ("social_policy_reliability_gated_demo_bias_observed_outcome_q_perseveration", "demo_bias_rel"),
        ("social_dirichlet_reliability_gated_demo_bias_observed_outcome_q_perseveration", "demo_bias_rel"),
        ("social_observed_outcome_policy_shared_mix", "mix_weight"),
        ("social_observed_outcome_policy_shared_mix_perseveration", "mix_weight"),
        ("social_observed_outcome_policy_independent_mix_perseveration", "beta_q"),
        ("social_policy_learning_only", "alpha_policy"),
        ("social_policy_learning_only_perseveration", "alpha_policy"),
        ("social_self_outcome_value_shaping", "alpha_self"),
    ],
)
def test_sample_subject_hierarchical_posterior_stan_supports_social_models(
    monkeypatch: pytest.MonkeyPatch,
    component_id: str,
    parameter_name: str,
) -> None:
    """Stan helper should accept all social model component IDs."""

    block = BlockData(block_id="b1", trials=_social_trials())
    subject = SubjectData(subject_id="s1", blocks=(block,))

    fake_fit = _fake_fit(n_draws=6, n_blocks=1, n_params=1, block_param_value=0.41)
    captured: dict[str, object] = {}

    def _fake_run(**kwargs: object) -> _FakeFit:
        captured.update(kwargs)
        return fake_fit

    monkeypatch.setattr(hierarchical_stan_module, "_run_stan_hierarchical_nuts", _fake_run)

    result = sample_subject_hierarchical_posterior_stan(
        subject,
        model_component_id=component_id,
        model_kwargs={},
        parameter_names=[parameter_name],
        n_samples=3,
        n_warmup=2,
        thin=1,
        n_chains=2,
        random_seed=11,
    )

    assert result.subject_id == "s1"
    assert result.parameter_names == (parameter_name,)
    assert len(result.draws) == 6
    assert result.draws[0].candidate.block_params[0][parameter_name] == pytest.approx(0.41)
    assert captured["cache_tag"] == f"hierarchical_{component_id}"

    stan_data = captured["stan_data"]
    assert isinstance(stan_data, dict)
    assert stan_data["actor_code"][0][0] == 2
    assert stan_data["actor_code"][0][1] == 1


def test_sample_subject_hierarchical_posterior_stan_rejects_unsupported_model() -> None:
    """Stan helper should fail fast on unsupported model component IDs."""

    block = BlockData(
        block_id="b1",
        trials=(_trial(0, 1, 1.0), _trial(1, 0, 0.0)),
    )
    subject = SubjectData(subject_id="s1", blocks=(block,))

    with pytest.raises(ValueError, match="does not support"):
        sample_subject_hierarchical_posterior_stan(
            subject,
            model_component_id="totally_unknown_component",
            model_kwargs={},
            parameter_names=["alpha"],
        )


def test_sample_subject_hierarchical_posterior_stan_supports_per_parameter_priors_asocial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asocial Stan helper should accept per-parameter prior mappings."""

    block = BlockData(
        block_id="b1",
        trials=(_trial(0, 1, 1.0), _trial(1, 0, 0.0)),
    )
    subject = SubjectData(subject_id="s1", blocks=(block,))
    fake_fit = _fake_fit(n_draws=4, n_blocks=1, n_params=2, block_param_value=0.3)
    captured: dict[str, object] = {}

    def _fake_run(**kwargs: object) -> _FakeFit:
        captured.update(kwargs)
        return fake_fit

    monkeypatch.setattr(hierarchical_stan_module, "_run_stan_hierarchical_nuts", _fake_run)
    sample_subject_hierarchical_posterior_stan(
        subject,
        model_component_id="asocial_state_q_value_softmax",
        model_kwargs={},
        parameter_names=["alpha", "beta"],
        mu_prior_mean={"alpha": 0.1, "beta": 0.2},
        mu_prior_std={"alpha": 1.1, "beta": 1.2},
        log_sigma_prior_mean={"alpha": -0.9, "beta": -0.8},
        log_sigma_prior_std={"alpha": 0.7, "beta": 0.8},
        n_samples=2,
        n_warmup=1,
        n_chains=2,
    )

    stan_data = captured["stan_data"]
    assert isinstance(stan_data, dict)
    assert stan_data["mu_prior_mean"] == pytest.approx([0.1, 0.2])
    assert stan_data["mu_prior_std"] == pytest.approx([1.1, 1.2])
    assert stan_data["log_sigma_prior_mean"] == pytest.approx([-0.9, -0.8])
    assert stan_data["log_sigma_prior_std"] == pytest.approx([0.7, 0.8])


def test_sample_subject_hierarchical_posterior_stan_supports_per_parameter_priors_social(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Social Stan helper should accept per-parameter prior mappings."""

    block = BlockData(block_id="b1", trials=_social_trials())
    subject = SubjectData(subject_id="s1", blocks=(block,))
    fake_fit = _fake_fit(n_draws=4, n_blocks=1, n_params=2, block_param_value=0.3)
    captured: dict[str, object] = {}

    def _fake_run(**kwargs: object) -> _FakeFit:
        captured.update(kwargs)
        return fake_fit

    monkeypatch.setattr(hierarchical_stan_module, "_run_stan_hierarchical_nuts", _fake_run)
    sample_subject_hierarchical_posterior_stan(
        subject,
        model_component_id="social_observed_outcome_q_perseveration",
        model_kwargs={},
        parameter_names=["alpha_observed", "beta"],
        mu_prior_mean={"alpha_observed": 0.4, "beta": 0.5},
        mu_prior_std={"alpha_observed": 1.4, "beta": 1.5},
        log_sigma_prior_mean={"alpha_observed": -0.4, "beta": -0.5},
        log_sigma_prior_std={"alpha_observed": 0.9, "beta": 1.0},
        n_samples=2,
        n_warmup=1,
        n_chains=2,
    )

    stan_data = captured["stan_data"]
    assert isinstance(stan_data, dict)
    assert stan_data["mu_prior_mean"] == pytest.approx([0.4, 0.5])
    assert stan_data["mu_prior_std"] == pytest.approx([1.4, 1.5])
    assert stan_data["log_sigma_prior_mean"] == pytest.approx([-0.4, -0.5])
    assert stan_data["log_sigma_prior_std"] == pytest.approx([0.9, 1.0])
