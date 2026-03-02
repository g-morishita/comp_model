"""Tests for config-driven model-comparison helpers."""

from __future__ import annotations

import pytest

import comp_model.inference.model_selection_config as model_selection_config_module
from comp_model.core.data import BlockData, StudyData, SubjectData, TrialDecision
from comp_model.demonstrators import FixedSequenceDemonstrator
from comp_model.inference import (
    compare_dataset_candidates_from_config,
    compare_study_candidates_from_config,
    compare_subject_candidates_from_config,
)
from comp_model.models import UniformRandomPolicyModel
from comp_model.problems import TwoStageSocialBanditProgram
from comp_model.runtime import SimulationConfig, run_trial_program


def _trial(trial_index: int, action: int, reward: float) -> TrialDecision:
    """Build one trial row for model-comparison config tests."""

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
    """Generate one two-actor social trace for model-selection config tests."""

    return run_trial_program(
        program=TwoStageSocialBanditProgram([0.5, 0.5]),
        models={
            "subject": UniformRandomPolicyModel(),
            "demonstrator": FixedSequenceDemonstrator(sequence=[1] * n_trials),
        },
        config=SimulationConfig(n_trials=n_trials, seed=seed),
    )


class _FakeJointResult:
    """Minimal subject-level fit result used for joint-comparison tests."""

    def __init__(self, *, log_likelihood: float, log_posterior: float, alpha: float) -> None:
        self.total_log_likelihood = float(log_likelihood)
        self.total_log_posterior = float(log_posterior)
        self.mean_map_params = {"alpha": float(alpha)}


def test_compare_dataset_candidates_from_config_supports_mle() -> None:
    """Config model-comparison should select stronger MLE candidate."""

    rows = tuple(_trial(index, action=1, reward=1.0) for index in range(12))
    config = {
        "criterion": "log_likelihood",
        "candidates": [
            {
                "name": "good_mle",
                "model": {
                    "component_id": "asocial_state_q_value_softmax",
                    "kwargs": {},
                },
                "estimator": {
                    "type": "mle", "solver": "grid_search",
                    "parameter_grid": {
                        "alpha": [0.8],
                        "beta": [8.0],
                        "initial_value": [0.0],
                    },
                },
                "n_parameters": 3,
            },
            {
                "name": "bad_mle",
                "model": {
                    "component_id": "asocial_state_q_value_softmax",
                    "kwargs": {},
                },
                "estimator": {
                    "type": "mle", "solver": "grid_search",
                    "parameter_grid": {
                        "alpha": [0.2],
                        "beta": [0.0],
                        "initial_value": [0.0],
                    },
                },
                "n_parameters": 3,
            },
        ],
    }

    result = compare_dataset_candidates_from_config(rows, config=config)
    assert result.selected_candidate_name == "good_mle"
    assert len(result.comparisons) == 2


def test_compare_dataset_candidates_from_config_rejects_removed_scipy_map() -> None:
    """Config model-comparison should reject removed SciPy MAP estimators."""

    rows = (_trial(0, action=1, reward=1.0),)
    config = {
        "candidates": [
            {
                "name": "map_candidate",
                "model": {
                    "component_id": "asocial_state_q_value_softmax",
                    "kwargs": {},
                },
                "prior": {
                    "type": "independent",
                    "parameters": {
                        "alpha": {"distribution": "uniform", "lower": 0.0, "upper": 1.0},
                    },
                },
                "estimator": {
                    "type": "scipy_map",
                    "initial_params": {"alpha": 0.5},
                    "bounds": {"alpha": [0.0, 1.0]},
                },
            },
        ],
    }

    with pytest.raises(ValueError, match="estimator.type must be one of"):
        compare_dataset_candidates_from_config(rows, config=config)


def test_compare_dataset_candidates_from_config_rejects_random_walk_estimator() -> None:
    """Config model-comparison should reject removed random-walk estimators."""

    rows = tuple(_trial(index, action=1, reward=1.0) for index in range(18))
    config = {
        "criterion": "waic",
        "candidates": [
            {
                "name": "good_mcmc",
                "model": {
                    "component_id": "asocial_state_q_value_softmax",
                    "kwargs": {},
                },
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
                    "initial_params": {"alpha": 0.6, "beta": 3.0, "initial_value": 0.0},
                    "n_samples": 20,
                    "n_warmup": 20,
                    "thin": 1,
                    "proposal_scales": {"alpha": 0.05, "beta": 0.2, "initial_value": 0.1},
                    "bounds": {
                        "alpha": [0.0, 1.0],
                        "beta": [0.0, 20.0],
                        "initial_value": [None, None],
                    },
                    "random_seed": 3,
                },
                "n_parameters": 3,
            },
            {
                "name": "bad_mcmc",
                "model": {
                    "component_id": "asocial_state_q_value_softmax",
                    "kwargs": {},
                },
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
                    "initial_params": {"alpha": 0.2, "beta": 0.1, "initial_value": 0.0},
                    "n_samples": 20,
                    "n_warmup": 20,
                    "thin": 1,
                    "proposal_scales": {"alpha": 0.02, "beta": 0.05, "initial_value": 0.05},
                    "bounds": {
                        "alpha": [0.0, 0.4],
                        "beta": [0.0, 0.5],
                        "initial_value": [None, None],
                    },
                    "random_seed": 4,
                },
                "n_parameters": 3,
            },
        ],
    }

    with pytest.raises(ValueError, match="estimator.type must be one of"):
        compare_dataset_candidates_from_config(rows, config=config)


def test_compare_subject_and_study_candidates_from_config() -> None:
    """Config comparison helpers should support subject and study datasets."""

    block = BlockData(
        block_id="b0",
        trials=tuple(_trial(index, action=1, reward=1.0) for index in range(6)),
    )
    subject = SubjectData(subject_id="s1", blocks=(block,))
    study = StudyData(subjects=(subject,))

    config = {
        "criterion": "log_likelihood",
        "candidates": [
            {
                "name": "good_mle",
                "model": {
                    "component_id": "asocial_state_q_value_softmax",
                    "kwargs": {},
                },
                "estimator": {
                    "type": "mle", "solver": "grid_search",
                    "parameter_grid": {
                        "alpha": [0.8],
                        "beta": [8.0],
                        "initial_value": [0.0],
                    },
                },
                "n_parameters": 3,
            },
            {
                "name": "bad_mle",
                "model": {
                    "component_id": "asocial_state_q_value_softmax",
                    "kwargs": {},
                },
                "estimator": {
                    "type": "mle", "solver": "grid_search",
                    "parameter_grid": {
                        "alpha": [0.2],
                        "beta": [0.0],
                        "initial_value": [0.0],
                    },
                },
                "n_parameters": 3,
            },
        ],
    }

    subject_result = compare_subject_candidates_from_config(subject, config=config)
    assert subject_result.selected_candidate_name == "good_mle"

    study_result = compare_study_candidates_from_config(study, config=config)
    assert study_result.selected_candidate_name == "good_mle"
    assert study_result.n_subjects == 1


def test_compare_subject_candidates_from_config_supports_hierarchical_stan_estimators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Subject config comparison should accept hierarchical Stan candidate estimators."""

    subject = SubjectData(
        subject_id="s1",
        blocks=(
            BlockData(
                block_id="b0",
                trials=tuple(_trial(index, action=1, reward=1.0) for index in range(4)),
            ),
            BlockData(
                block_id="b1",
                trials=tuple(_trial(index, action=1, reward=1.0) for index in range(4)),
            ),
        ),
    )
    config = {
        "criterion": "log_likelihood",
        "block_fit_strategy": "joint",
        "candidates": [
            {
                "name": "hier_good",
                "model": {
                    "component_id": "asocial_state_q_value_softmax",
                    "kwargs": {"beta": 2.0, "initial_value": 0.0},
                },
                "estimator": {
                    "type": "within_subject_hierarchical_stan_nuts",
                    "parameter_names": ["alpha"],
                    "transforms": {"alpha": "unit_interval_logit"},
                    "n_samples": 4,
                    "n_warmup": 2,
                    "n_chains": 2,
                },
                "n_parameters": 1,
            },
            {
                "name": "hier_bad",
                "model": {
                    "component_id": "asocial_state_q_value_softmax",
                    "kwargs": {"beta": 0.2, "initial_value": 0.0},
                },
                "estimator": {
                    "type": "within_subject_hierarchical_stan_nuts",
                    "parameter_names": ["alpha"],
                    "transforms": {"alpha": "unit_interval_logit"},
                    "n_samples": 4,
                    "n_warmup": 2,
                    "n_chains": 2,
                },
                "n_parameters": 1,
            },
        ],
    }

    def _fake_fit_subject_auto_from_config(
        subject_data: SubjectData,
        *,
        config: dict,
        **kwargs: object,
    ) -> object:
        assert subject_data.subject_id == "s1"
        beta = float(config["model"]["kwargs"]["beta"])
        if beta > 1.0:
            return _FakeJointResult(log_likelihood=-5.0, log_posterior=-5.5, alpha=0.3)
        return _FakeJointResult(log_likelihood=-9.0, log_posterior=-9.5, alpha=0.3)

    monkeypatch.setattr(
        model_selection_config_module,
        "fit_subject_auto_from_config",
        _fake_fit_subject_auto_from_config,
    )

    result = compare_subject_candidates_from_config(subject, config=config)
    assert result.selected_candidate_name == "hier_good"


def test_compare_subject_candidates_from_config_supports_joint_block_strategy() -> None:
    """Subject/study config comparison should support joint block fitting mode."""

    subject = SubjectData(
        subject_id="s1",
        blocks=(
            BlockData(
                block_id="b0",
                trials=tuple(_trial(index, action=1, reward=1.0) for index in range(6)),
            ),
            BlockData(
                block_id="b1",
                trials=tuple(_trial(index, action=1, reward=1.0) for index in range(6)),
            ),
        ),
    )
    study = StudyData(subjects=(subject,))
    config = {
        "criterion": "log_likelihood",
        "block_fit_strategy": "joint",
        "candidates": [
            {
                "name": "good_mle",
                "model": {
                    "component_id": "asocial_state_q_value_softmax",
                    "kwargs": {},
                },
                "estimator": {
                    "type": "mle", "solver": "grid_search",
                    "parameter_grid": {
                        "alpha": [0.8],
                        "beta": [8.0],
                        "initial_value": [0.0],
                    },
                },
                "n_parameters": 3,
            },
            {
                "name": "bad_mle",
                "model": {
                    "component_id": "asocial_state_q_value_softmax",
                    "kwargs": {},
                },
                "estimator": {
                    "type": "mle", "solver": "grid_search",
                    "parameter_grid": {
                        "alpha": [0.2],
                        "beta": [0.0],
                        "initial_value": [0.0],
                    },
                },
                "n_parameters": 3,
            },
        ],
    }

    subject_result = compare_subject_candidates_from_config(subject, config=config)
    assert subject_result.selected_candidate_name == "good_mle"

    study_result = compare_study_candidates_from_config(study, config=config)
    assert study_result.selected_candidate_name == "good_mle"


def test_compare_dataset_candidates_from_config_rejects_unknown_top_level_keys() -> None:
    """Model-selection config should fail fast on unknown top-level keys."""

    rows = (_trial(0, action=1, reward=1.0),)
    config = {
        "candidates": [
            {
                "name": "mle",
                "model": {"component_id": "asocial_state_q_value_softmax", "kwargs": {}},
                "estimator": {
                    "type": "mle", "solver": "grid_search",
                    "parameter_grid": {"alpha": [0.5], "beta": [1.0], "initial_value": [0.0]},
                },
            },
        ],
        "unexpected": True,
    }

    with pytest.raises(ValueError, match="config has unknown keys"):
        compare_dataset_candidates_from_config(rows, config=config)


def test_compare_dataset_candidates_from_config_rejects_unknown_candidate_keys() -> None:
    """Candidate parser should reject unknown keys in candidate entries."""

    rows = (_trial(0, action=1, reward=1.0),)
    config = {
        "candidates": [
            {
                "name": "mle",
                "model": {"component_id": "asocial_state_q_value_softmax", "kwargs": {}},
                "estimator": {
                    "type": "mle", "solver": "grid_search",
                    "parameter_grid": {"alpha": [0.5], "beta": [1.0], "initial_value": [0.0]},
                },
                "oops": 1,
            },
        ],
    }

    with pytest.raises(ValueError, match="config.candidates\\[0\\] has unknown keys"):
        compare_dataset_candidates_from_config(rows, config=config)


def test_compare_dataset_candidates_from_config_supports_candidate_likelihood_config() -> None:
    """Candidate entries should accept per-candidate likelihood configs."""

    trace = _social_trace(n_trials=10, seed=12)
    config = {
        "criterion": "log_likelihood",
        "candidates": [
            {
                "name": "candidate_a",
                "model": {
                    "component_id": "asocial_state_q_value_softmax",
                    "kwargs": {},
                },
                "estimator": {
                    "type": "mle", "solver": "grid_search",
                    "parameter_grid": {
                        "alpha": [0.2],
                        "beta": [1.0],
                        "initial_value": [0.0],
                    },
                },
                "likelihood": {
                    "type": "actor_subset_replay",
                    "fitted_actor_id": "subject",
                    "scored_actor_ids": ["subject"],
                    "auto_fill_unmodeled_actors": True,
                },
                "n_parameters": 3,
            },
            {
                "name": "candidate_b",
                "model": {
                    "component_id": "asocial_state_q_value_softmax",
                    "kwargs": {},
                },
                "estimator": {
                    "type": "mle", "solver": "grid_search",
                    "parameter_grid": {
                        "alpha": [0.8],
                        "beta": [6.0],
                        "initial_value": [0.0],
                    },
                },
                "likelihood": {
                    "type": "actor_subset_replay",
                    "fitted_actor_id": "subject",
                    "scored_actor_ids": ["subject"],
                    "auto_fill_unmodeled_actors": True,
                },
                "n_parameters": 3,
            },
        ],
    }

    result = compare_dataset_candidates_from_config(trace, config=config)
    assert len(result.comparisons) == 2
    assert {item.candidate_name for item in result.comparisons} == {"candidate_a", "candidate_b"}
