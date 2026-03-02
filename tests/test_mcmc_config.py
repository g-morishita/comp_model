"""Tests for Stan hierarchical config sampling helpers."""

from __future__ import annotations

import pytest

import comp_model.inference.mcmc_config as mcmc_config_module
from comp_model.core.data import BlockData, StudyData, SubjectData, TrialDecision
from comp_model.inference import (
    fit_study_auto_from_config,
    fit_subject_auto_from_config,
    hierarchical_stan_estimator_spec_from_config,
    sample_study_hierarchical_posterior_from_config,
    sample_subject_hierarchical_posterior_from_config,
)


def _trial(trial_index: int, action: int, reward: float) -> TrialDecision:
    """Build one trial row for Stan config tests."""

    return TrialDecision(
        trial_index=trial_index,
        decision_index=0,
        actor_id="subject",
        available_actions=(0, 1),
        action=action,
        observation={"state": 0},
        outcome={"reward": reward},
    )


def _hierarchical_stan_config() -> dict:
    """Build one minimal hierarchical Stan NUTS config."""

    return {
        "model": {
            "component_id": "asocial_state_q_value_softmax",
            "kwargs": {
                "beta": 2.0,
                "initial_value": 0.0,
            },
        },
        "estimator": {
            "type": "within_subject_hierarchical_stan_nuts",
            "parameter_names": ["alpha"],
            "transforms": {"alpha": "unit_interval_logit"},
            "initial_group_location": {"alpha": 0.5},
            "initial_group_scale": {"alpha": 0.5},
            "n_samples": 10,
            "n_warmup": 8,
            "thin": 1,
            "n_chains": 2,
            "parallel_chains": 2,
            "adapt_delta": 0.9,
            "max_treedepth": 10,
            "refresh": 0,
            "random_seed": 7,
        },
    }


def test_hierarchical_stan_estimator_spec_from_config_parses_fields() -> None:
    """Hierarchical Stan parser should construct a full spec."""

    spec = hierarchical_stan_estimator_spec_from_config(_hierarchical_stan_config()["estimator"])
    assert spec.parameter_names == ("alpha",)
    assert spec.transform_kinds == {"alpha": "unit_interval_logit"}
    assert spec.initial_group_location == {"alpha": pytest.approx(0.5)}
    assert spec.initial_group_scale == {"alpha": pytest.approx(0.5)}
    assert spec.n_samples == 10
    assert spec.n_warmup == 8
    assert spec.thin == 1
    assert spec.n_chains == 2
    assert spec.parallel_chains == 2
    assert spec.adapt_delta == pytest.approx(0.9)
    assert spec.max_treedepth == 10
    assert spec.random_seed == 7


def test_hierarchical_stan_estimator_spec_from_config_supports_prior_mappings() -> None:
    """Hierarchical Stan parser should accept per-parameter prior mappings."""

    estimator_cfg = _hierarchical_stan_config()["estimator"]
    estimator_cfg["parameter_names"] = ["alpha", "beta"]
    estimator_cfg["mu_prior_mean"] = {"alpha": 0.1, "beta": 0.2}
    estimator_cfg["mu_prior_std"] = {"alpha": 1.1, "beta": 1.2}
    estimator_cfg["log_sigma_prior_mean"] = {"alpha": -0.9, "beta": -0.8}
    estimator_cfg["log_sigma_prior_std"] = {"alpha": 0.7, "beta": 0.8}

    spec = hierarchical_stan_estimator_spec_from_config(estimator_cfg)
    assert spec.parameter_names == ("alpha", "beta")
    assert spec.mu_prior_mean == {"alpha": pytest.approx(0.1), "beta": pytest.approx(0.2)}
    assert spec.mu_prior_std == {"alpha": pytest.approx(1.1), "beta": pytest.approx(1.2)}
    assert spec.log_sigma_prior_mean == {"alpha": pytest.approx(-0.9), "beta": pytest.approx(-0.8)}
    assert spec.log_sigma_prior_std == {"alpha": pytest.approx(0.7), "beta": pytest.approx(0.8)}


def test_hierarchical_stan_estimator_spec_from_config_rejects_unknown_keys() -> None:
    """Hierarchical Stan parser should reject unknown keys."""

    estimator_cfg = dict(_hierarchical_stan_config()["estimator"])
    estimator_cfg["unexpected"] = 1
    with pytest.raises(ValueError, match="estimator has unknown keys"):
        hierarchical_stan_estimator_spec_from_config(estimator_cfg)


def test_sample_hierarchical_stan_posterior_subject_study_from_config_dispatches() -> None:
    """Hierarchical config runner should route Stan estimator type."""

    block_1 = BlockData(
        block_id="b1",
        trials=(_trial(0, 1, 1.0), _trial(1, 0, 0.0), _trial(2, 1, 1.0)),
    )
    block_2 = BlockData(
        block_id="b2",
        trials=(_trial(0, 0, 0.0), _trial(1, 1, 1.0), _trial(2, 1, 1.0)),
    )
    subject = SubjectData(subject_id="s1", blocks=(block_1, block_2))
    study = StudyData(subjects=(subject,))

    sentinel_subject = object()
    sentinel_study = object()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        mcmc_config_module,
        "sample_subject_hierarchical_posterior_stan",
        lambda *args, **kwargs: sentinel_subject,
    )
    monkeypatch.setattr(
        mcmc_config_module,
        "sample_study_hierarchical_posterior_stan",
        lambda *args, **kwargs: sentinel_study,
    )
    try:
        subject_result = sample_subject_hierarchical_posterior_from_config(
            subject,
            config=_hierarchical_stan_config(),
        )
        assert subject_result is sentinel_subject

        study_result = sample_study_hierarchical_posterior_from_config(
            study,
            config=_hierarchical_stan_config(),
        )
        assert study_result is sentinel_study
    finally:
        monkeypatch.undo()


def test_fit_auto_dispatches_hierarchical_stan_for_subject_and_study() -> None:
    """Auto-dispatch should route hierarchical Stan estimator type."""

    block_1 = BlockData(
        block_id="b1",
        trials=(_trial(0, 1, 1.0), _trial(1, 0, 0.0), _trial(2, 1, 1.0)),
    )
    block_2 = BlockData(
        block_id="b2",
        trials=(_trial(0, 0, 0.0), _trial(1, 1, 1.0), _trial(2, 1, 1.0)),
    )
    subject = SubjectData(subject_id="s1", blocks=(block_1, block_2))
    study = StudyData(subjects=(subject,))

    sentinel_subject = object()
    sentinel_study = object()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        mcmc_config_module,
        "sample_subject_hierarchical_posterior_stan",
        lambda *args, **kwargs: sentinel_subject,
    )
    monkeypatch.setattr(
        mcmc_config_module,
        "sample_study_hierarchical_posterior_stan",
        lambda *args, **kwargs: sentinel_study,
    )
    try:
        subject_result = fit_subject_auto_from_config(
            subject,
            config=_hierarchical_stan_config(),
        )
        assert subject_result is sentinel_subject

        study_result = fit_study_auto_from_config(
            study,
            config=_hierarchical_stan_config(),
        )
        assert study_result is sentinel_study
    finally:
        monkeypatch.undo()
