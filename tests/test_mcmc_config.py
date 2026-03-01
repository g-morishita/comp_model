"""Tests for config-driven MCMC posterior sampling helpers."""

from __future__ import annotations

import pytest

import comp_model.inference.mcmc_config as mcmc_config_module
from comp_model.core.data import BlockData, StudyData, SubjectData, TrialDecision
from comp_model.demonstrators import FixedSequenceDemonstrator
from comp_model.inference import (
    fit_block_auto_from_config,
    fit_dataset_auto_from_config,
    fit_study_auto_from_config,
    fit_subject_auto_from_config,
    hierarchical_mcmc_estimator_spec_from_config,
    hierarchical_stan_estimator_spec_from_config,
    mcmc_estimator_spec_from_config,
    sample_study_hierarchical_posterior_from_config,
    sample_posterior_block_from_config,
    sample_posterior_dataset_from_config,
    sample_posterior_study_from_config,
    sample_posterior_subject_from_config,
    sample_subject_hierarchical_posterior_from_config,
)
from comp_model.models import UniformRandomPolicyModel
from comp_model.problems import TwoStageSocialBanditProgram
from comp_model.runtime import SimulationConfig, run_trial_program


def _trial(trial_index: int, action: int, reward: float) -> TrialDecision:
    """Build one trial row for MCMC config tests."""

    return TrialDecision(
        trial_index=trial_index,
        decision_index=0,
        actor_id="subject",
        available_actions=(0, 1),
        action=action,
        observation={"state": 0},
        outcome={"reward": reward},
    )


def _social_trace(*, n_trials: int, seed: int):
    """Generate one two-actor social trace for MCMC config tests."""

    return run_trial_program(
        program=TwoStageSocialBanditProgram([0.5, 0.5]),
        models={
            "subject": UniformRandomPolicyModel(),
            "demonstrator": FixedSequenceDemonstrator(sequence=[1] * n_trials),
        },
        config=SimulationConfig(n_trials=n_trials, seed=seed),
    )


def _mcmc_config() -> dict:
    """Build one minimal MCMC config."""

    return {
        "model": {"component_id": "asocial_state_q_value_softmax", "kwargs": {}},
        "prior": {
            "type": "independent",
            "parameters": {
                "alpha": {"distribution": "uniform", "lower": 0.0, "upper": 1.0},
                "beta": {"distribution": "uniform", "lower": 0.0, "upper": 20.0},
                "initial_value": {"distribution": "normal", "mean": 0.0, "std": 1.0},
            },
        },
        "estimator": {
            "type": "random_walk_metropolis",
            "initial_params": {"alpha": 0.4, "beta": 2.0, "initial_value": 0.0},
            "n_samples": 20,
            "n_warmup": 20,
            "thin": 1,
            "proposal_scales": {"alpha": 0.05, "beta": 0.2, "initial_value": 0.1},
            "bounds": {
                "alpha": [0.0, 1.0],
                "beta": [0.0, 20.0],
                "initial_value": [None, None],
            },
            "random_seed": 9,
        },
    }


def _hierarchical_mcmc_config() -> dict:
    """Build one minimal hierarchical MCMC config."""

    return {
        "model": {
            "component_id": "asocial_state_q_value_softmax",
            "kwargs": {
                "beta": 2.0,
                "initial_value": 0.0,
            },
        },
        "estimator": {
            "type": "within_subject_hierarchical_random_walk_metropolis",
            "parameter_names": ["alpha"],
            "transforms": {"alpha": "unit_interval_logit"},
            "initial_group_location": {"alpha": 0.5},
            "initial_group_scale": {"alpha": 0.5},
            "n_samples": 12,
            "n_warmup": 8,
            "thin": 1,
            "proposal_scale_group_location": 0.08,
            "proposal_scale_group_log_scale": 0.05,
            "proposal_scale_block_z": 0.08,
            "random_seed": 7,
        },
    }


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


def test_mcmc_estimator_spec_from_config_parses_fields() -> None:
    """MCMC estimator parser should construct a full spec."""

    spec = mcmc_estimator_spec_from_config(_mcmc_config()["estimator"])
    assert spec.n_samples == 20
    assert spec.n_warmup == 20
    assert spec.thin == 1
    assert spec.initial_params["alpha"] == pytest.approx(0.4)
    assert spec.proposal_scales is not None
    assert spec.proposal_scales["beta"] == pytest.approx(0.2)
    assert spec.bounds is not None
    assert spec.bounds["alpha"] == (0.0, 1.0)
    assert spec.random_seed == 9


def test_mcmc_estimator_spec_from_config_rejects_unknown_keys() -> None:
    """MCMC estimator parser should reject unknown keys."""

    estimator_cfg = dict(_mcmc_config()["estimator"])
    estimator_cfg["unexpected"] = 1
    with pytest.raises(ValueError, match="estimator has unknown keys"):
        mcmc_estimator_spec_from_config(estimator_cfg)


def test_sample_posterior_dataset_from_config_runs_end_to_end() -> None:
    """Config-driven MCMC helper should return posterior samples."""

    rows = (_trial(0, 1, 1.0), _trial(1, 0, 0.0), _trial(2, 1, 1.0), _trial(3, 1, 1.0))
    result = sample_posterior_dataset_from_config(rows, config=_mcmc_config())

    assert result.posterior_samples.n_draws == 20
    assert set(result.posterior_samples.parameter_names) == {
        "alpha",
        "beta",
        "initial_value",
    }


def test_sample_posterior_dataset_from_config_rejects_unknown_top_level_keys() -> None:
    """MCMC dataset config should fail fast on unknown top-level keys."""

    rows = (_trial(0, 1, 1.0), _trial(1, 0, 0.0))
    config = _mcmc_config()
    config["typo"] = True
    with pytest.raises(ValueError, match="config has unknown keys"):
        sample_posterior_dataset_from_config(rows, config=config)


def test_fit_dataset_auto_dispatches_mcmc() -> None:
    """Dataset auto-dispatch should route to MCMC posterior sampling."""

    rows = (_trial(0, 1, 1.0), _trial(1, 0, 0.0), _trial(2, 1, 1.0))
    result = fit_dataset_auto_from_config(rows, config=_mcmc_config())
    assert result.posterior_samples.n_draws == 20


def test_sample_posterior_block_subject_study_from_config() -> None:
    """Config-driven MCMC helpers should support block/subject/study inputs."""

    block = BlockData(
        block_id="b0",
        trials=(_trial(0, 1, 1.0), _trial(1, 0, 0.0), _trial(2, 1, 1.0)),
    )
    subject = SubjectData(subject_id="s1", blocks=(block,))
    study = StudyData(subjects=(subject,))

    block_result = sample_posterior_block_from_config(block, config=_mcmc_config())
    assert block_result.n_trials == 3
    assert block_result.posterior_result.posterior_samples.n_draws == 20

    subject_result = sample_posterior_subject_from_config(subject, config=_mcmc_config())
    assert subject_result.subject_id == "s1"
    assert len(subject_result.block_results) == 1
    assert set(subject_result.mean_block_map_params) == {"alpha", "beta", "initial_value"}

    study_result = sample_posterior_study_from_config(study, config=_mcmc_config())
    assert study_result.n_subjects == 1
    assert len(study_result.subject_results) == 1


def test_fit_auto_dispatches_mcmc_for_all_dataset_levels() -> None:
    """Auto-dispatch should route MCMC estimator type across all levels."""

    block = BlockData(
        block_id="b0",
        trials=(_trial(0, 1, 1.0), _trial(1, 0, 0.0), _trial(2, 1, 1.0)),
    )
    subject = SubjectData(subject_id="s1", blocks=(block,))
    study = StudyData(subjects=(subject,))

    block_result = fit_block_auto_from_config(block, config=_mcmc_config())
    assert block_result.posterior_result.posterior_samples.n_draws == 20

    subject_result = fit_subject_auto_from_config(subject, config=_mcmc_config())
    assert len(subject_result.block_results) == 1

    study_result = fit_study_auto_from_config(study, config=_mcmc_config())
    assert study_result.n_subjects == 1


def test_sample_posterior_dataset_from_config_supports_social_actor_subset_likelihood() -> None:
    """MCMC config runner should parse actor-subset likelihood on social traces."""

    trace = _social_trace(n_trials=12, seed=11)
    config = _mcmc_config()
    config["likelihood"] = {
        "type": "actor_subset_replay",
        "fitted_actor_id": "subject",
        "scored_actor_ids": ["subject"],
        "auto_fill_unmodeled_actors": True,
    }

    result = sample_posterior_dataset_from_config(trace, config=config)
    assert result.posterior_samples.n_draws == 20


def test_hierarchical_mcmc_estimator_spec_from_config_parses_fields() -> None:
    """Hierarchical MCMC parser should construct a full spec."""

    spec = hierarchical_mcmc_estimator_spec_from_config(_hierarchical_mcmc_config()["estimator"])
    assert spec.parameter_names == ("alpha",)
    assert spec.transforms is not None
    assert spec.initial_group_location == {"alpha": pytest.approx(0.5)}
    assert spec.initial_group_scale == {"alpha": pytest.approx(0.5)}
    assert spec.n_samples == 12
    assert spec.n_warmup == 8
    assert spec.thin == 1
    assert spec.random_seed == 7


def test_hierarchical_mcmc_estimator_spec_from_config_rejects_unknown_keys() -> None:
    """Hierarchical MCMC parser should reject unknown keys."""

    estimator_cfg = dict(_hierarchical_mcmc_config()["estimator"])
    estimator_cfg["unexpected"] = 1
    with pytest.raises(ValueError, match="estimator has unknown keys"):
        hierarchical_mcmc_estimator_spec_from_config(estimator_cfg)


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


def test_sample_hierarchical_posterior_subject_study_from_config() -> None:
    """Hierarchical MCMC config helpers should support subject/study inputs."""

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

    subject_result = sample_subject_hierarchical_posterior_from_config(
        subject,
        config=_hierarchical_mcmc_config(),
    )
    assert subject_result.subject_id == "s1"
    assert len(subject_result.draws) == 12

    study_result = sample_study_hierarchical_posterior_from_config(
        study,
        config=_hierarchical_mcmc_config(),
    )
    assert study_result.n_subjects == 1
    assert len(study_result.subject_results[0].draws) == 12


def test_fit_auto_dispatches_hierarchical_mcmc_for_subject_and_study() -> None:
    """Auto-dispatch should route hierarchical MCMC estimator type."""

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

    subject_result = fit_subject_auto_from_config(
        subject,
        config=_hierarchical_mcmc_config(),
    )
    assert len(subject_result.draws) == 12

    study_result = fit_study_auto_from_config(
        study,
        config=_hierarchical_mcmc_config(),
    )
    assert study_result.n_subjects == 1


def test_sample_subject_hierarchical_from_config_rejects_unknown_top_level_keys() -> None:
    """Hierarchical MCMC config runner should fail fast on unknown keys."""

    subject = SubjectData(
        subject_id="s1",
        blocks=(
            BlockData(
                block_id="b1",
                trials=(_trial(0, 1, 1.0), _trial(1, 0, 0.0), _trial(2, 1, 1.0)),
            ),
            BlockData(
                block_id="b2",
                trials=(_trial(0, 0, 0.0), _trial(1, 1, 1.0), _trial(2, 1, 1.0)),
            ),
        ),
    )
    config = _hierarchical_mcmc_config()
    config["typo"] = True
    with pytest.raises(ValueError, match="config has unknown keys"):
        sample_subject_hierarchical_posterior_from_config(subject, config=config)
