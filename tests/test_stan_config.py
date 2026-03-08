"""Tests for explicit Stan config helpers."""

from __future__ import annotations

import pytest

import comp_model.inference.stan_config as stan_config_module
from comp_model.core.data import BlockData, StudyData, SubjectData, TrialDecision
from comp_model.inference import (
    fit_study_auto_from_config,
    fit_subject_auto_from_config,
    infer_study_stan_from_config,
    infer_subject_stan_from_config,
    stan_estimator_spec_from_config,
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


def _subject_shared_nuts_config() -> dict:
    """Build one minimal subject-shared Stan NUTS config."""

    return {
        "model": {
            "component_id": "asocial_state_q_value_softmax",
            "kwargs": {"beta": 2.0, "initial_value": 0.0},
        },
        "estimator": {
            "type": "subject_shared_stan_nuts",
            "parameter_names": ["alpha"],
            "transforms": {"alpha": "unit_interval_logit"},
            "initial_group_location": {"alpha": 0.5},
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


def _subject_block_map_config() -> dict:
    """Build one minimal subject -> block Stan MAP config."""

    return {
        "model": {
            "component_id": "asocial_state_q_value_softmax",
            "kwargs": {"initial_value": 0.0},
        },
        "estimator": {
            "type": "subject_block_hierarchy_stan_map",
            "parameter_names": ["alpha", "beta"],
            "transforms": {"alpha": "unit_interval_logit", "beta": "positive_log"},
            "initial_group_location": {"alpha": 0.5, "beta": 1.0},
            "initial_group_scale": {"alpha": 0.5, "beta": 0.5},
            "method": "lbfgs",
            "max_iterations": 80,
            "jacobian": False,
            "random_seed": 7,
            "refresh": 0,
        },
    }


def _study_subject_nuts_config() -> dict:
    """Build one minimal population -> subject Stan NUTS config."""

    cfg = _subject_shared_nuts_config()
    cfg["estimator"] = dict(cfg["estimator"])
    cfg["estimator"]["type"] = "study_subject_hierarchy_stan_nuts"
    return cfg


def _study_subject_block_map_config() -> dict:
    """Build one minimal population -> subject -> block Stan MAP config."""

    cfg = _subject_block_map_config()
    cfg["estimator"] = dict(cfg["estimator"])
    cfg["estimator"]["type"] = "study_subject_block_hierarchy_stan_map"
    return cfg


def test_stan_estimator_spec_from_config_parses_subject_and_study_fields() -> None:
    """Stan parser should construct specs for the explicit estimator matrix."""

    subject_spec = stan_estimator_spec_from_config(_subject_shared_nuts_config()["estimator"])
    assert subject_spec.estimator_type == "subject_shared_stan_nuts"
    assert subject_spec.parameter_names == ("alpha",)
    assert subject_spec.transform_kinds == {"alpha": "unit_interval_logit"}
    assert subject_spec.n_samples == 10
    assert subject_spec.n_chains == 2

    study_spec = stan_estimator_spec_from_config(_study_subject_block_map_config()["estimator"])
    assert study_spec.estimator_type == "study_subject_block_hierarchy_stan_map"
    assert study_spec.parameter_names == ("alpha", "beta")
    assert study_spec.method == "lbfgs"
    assert study_spec.max_iterations == 80


def test_stan_estimator_spec_from_config_supports_prior_mappings() -> None:
    """Stan parser should accept per-parameter prior mappings."""

    estimator_cfg = _subject_block_map_config()["estimator"]
    estimator_cfg["mu_prior_mean"] = {"alpha": 0.1, "beta": 0.2}
    estimator_cfg["mu_prior_std"] = {"alpha": 1.1, "beta": 1.2}
    estimator_cfg["log_sigma_prior_mean"] = {"alpha": -0.9, "beta": -0.8}
    estimator_cfg["log_sigma_prior_std"] = {"alpha": 0.7, "beta": 0.8}

    spec = stan_estimator_spec_from_config(estimator_cfg)
    assert spec.mu_prior_mean == {"alpha": pytest.approx(0.1), "beta": pytest.approx(0.2)}
    assert spec.log_sigma_prior_std == {"alpha": pytest.approx(0.7), "beta": pytest.approx(0.8)}


def test_stan_estimator_spec_from_config_rejects_unknown_keys() -> None:
    """Stan parser should reject unknown keys."""

    estimator_cfg = dict(_subject_shared_nuts_config()["estimator"])
    estimator_cfg["unexpected"] = 1
    with pytest.raises(ValueError, match="estimator has unknown keys"):
        stan_estimator_spec_from_config(estimator_cfg)


def test_infer_subject_stan_from_config_dispatches() -> None:
    """Subject config runner should dispatch to the matching subject-level Stan helper."""

    subject = SubjectData(
        subject_id="s1",
        blocks=(BlockData(block_id="b1", trials=(_trial(0, 1, 1.0), _trial(1, 0, 0.0))),),
    )
    sentinel_shared = object()
    sentinel_block = object()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        stan_config_module,
        "draw_subject_shared_posterior_stan",
        lambda *args, **kwargs: sentinel_shared,
    )
    monkeypatch.setattr(
        stan_config_module,
        "estimate_subject_block_hierarchy_map_stan",
        lambda *args, **kwargs: sentinel_block,
    )
    try:
        shared_result = infer_subject_stan_from_config(subject, config=_subject_shared_nuts_config())
        assert shared_result is sentinel_shared

        block_result = infer_subject_stan_from_config(subject, config=_subject_block_map_config())
        assert block_result is sentinel_block
    finally:
        monkeypatch.undo()


def test_infer_study_stan_from_config_dispatches() -> None:
    """Study config runner should dispatch to the matching study-level Stan helper."""

    subject = SubjectData(
        subject_id="s1",
        blocks=(BlockData(block_id="b1", trials=(_trial(0, 1, 1.0), _trial(1, 0, 0.0))),),
    )
    study = StudyData(subjects=(subject,))

    sentinel_subject = object()
    sentinel_subject_block = object()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        stan_config_module,
        "draw_study_subject_hierarchy_posterior_stan",
        lambda *args, **kwargs: sentinel_subject,
    )
    monkeypatch.setattr(
        stan_config_module,
        "estimate_study_subject_block_hierarchy_map_stan",
        lambda *args, **kwargs: sentinel_subject_block,
    )
    try:
        subject_result = infer_study_stan_from_config(study, config=_study_subject_nuts_config())
        assert subject_result is sentinel_subject

        subject_block_result = infer_study_stan_from_config(study, config=_study_subject_block_map_config())
        assert subject_block_result is sentinel_subject_block
    finally:
        monkeypatch.undo()


def test_fit_auto_dispatches_explicit_stan_estimators() -> None:
    """Auto-dispatch should route subject and study Stan estimator types."""

    subject = SubjectData(
        subject_id="s1",
        blocks=(BlockData(block_id="b1", trials=(_trial(0, 1, 1.0), _trial(1, 0, 0.0))),),
    )
    study = StudyData(subjects=(subject,))

    sentinel_subject = object()
    sentinel_study = object()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        stan_config_module,
        "draw_subject_shared_posterior_stan",
        lambda *args, **kwargs: sentinel_subject,
    )
    monkeypatch.setattr(
        stan_config_module,
        "draw_study_subject_hierarchy_posterior_stan",
        lambda *args, **kwargs: sentinel_study,
    )
    try:
        subject_result = fit_subject_auto_from_config(subject, config=_subject_shared_nuts_config())
        assert subject_result is sentinel_subject

        study_result = fit_study_auto_from_config(study, config=_study_subject_nuts_config())
        assert study_result is sentinel_study
    finally:
        monkeypatch.undo()

