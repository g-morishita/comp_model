"""Tests for explicit Stan Bayesian hierarchy helpers."""

from __future__ import annotations

import numpy as np
import pytest

import comp_model.inference.hierarchical_stan as hierarchical_stan_module
from comp_model.core.data import BlockData, StudyData, SubjectData, TrialDecision
from comp_model.inference.hierarchical_stan import (
    draw_study_subject_block_hierarchy_posterior_stan,
    draw_study_subject_hierarchy_posterior_stan,
    draw_subject_block_hierarchy_posterior_stan,
    draw_subject_shared_posterior_stan,
    estimate_study_subject_block_hierarchy_map_stan,
    estimate_subject_shared_map_stan,
)
from comp_model.inference.hierarchical_stan_social import (
    load_social_stan_code,
    social_supported_component_ids,
)


def _trial(trial_index: int, action: int, reward: float) -> TrialDecision:
    """Build one trial row for Stan hierarchy tests."""

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
            learner_ids=("subject",),
            available_actions=(0, 1),
            action=1,
            observation={"state": 0, "stage": "demonstrator"},
            outcome={"reward": 1.0, "source_actor_id": "demonstrator"},
        ),
        TrialDecision(
            trial_index=0,
            decision_index=1,
            actor_id="subject",
            learner_ids=("subject",),
            available_actions=(0, 1),
            action=0,
            observation={"state": 0, "stage": "subject", "demonstrator_action": 1},
            outcome={"reward": 0.0, "source_actor_id": "subject"},
        ),
        TrialDecision(
            trial_index=1,
            decision_index=0,
            actor_id="demonstrator",
            learner_ids=("subject",),
            available_actions=(0, 1),
            action=0,
            observation={"state": 0, "stage": "demonstrator"},
            outcome={"reward": 0.0, "source_actor_id": "demonstrator"},
        ),
        TrialDecision(
            trial_index=1,
            decision_index=1,
            actor_id="subject",
            learner_ids=("subject",),
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


def _fake_subject_shared_fit(*, n_draws: int, n_blocks: int, n_params: int) -> _FakeFit:
    """Construct a fake CmdStan fit object for the subject-shared case."""

    return _FakeFit(
        variables={
            "subject_param_z": np.linspace(-0.2, 0.3, n_draws * n_params).reshape(n_draws, n_params),
            "block_z": np.full((n_draws, n_blocks, n_params), 0.1, dtype=float),
            "block_param": np.full((n_draws, n_blocks, n_params), 0.52, dtype=float),
            "log_likelihood_total": np.linspace(-5.0, -4.0, n_draws),
            "log_prior_total": np.linspace(-1.0, -0.5, n_draws),
            "log_posterior_total": np.linspace(-6.0, -4.5, n_draws),
        }
    )


def _fake_subject_block_fit(*, n_draws: int, n_blocks: int, n_params: int) -> _FakeFit:
    """Construct a fake CmdStan fit object for the subject -> block case."""

    return _FakeFit(
        variables={
            "subject_loc_z": np.linspace(-0.1, 0.2, n_draws * n_params).reshape(n_draws, n_params),
            "subject_log_scale": np.linspace(-1.0, -0.7, n_draws * n_params).reshape(n_draws, n_params),
            "block_z": np.full((n_draws, n_blocks, n_params), 0.15, dtype=float),
            "block_param": np.full((n_draws, n_blocks, n_params), 0.37, dtype=float),
            "log_likelihood_total": np.linspace(-7.0, -6.0, n_draws),
            "log_prior_total": np.linspace(-1.5, -0.8, n_draws),
            "log_posterior_total": np.linspace(-8.5, -6.8, n_draws),
        }
    )


def _fake_study_subject_fit(
    *,
    n_draws: int,
    n_subjects: int,
    n_blocks: int,
    n_params: int,
) -> _FakeFit:
    """Construct a fake CmdStan fit object for the population -> subject case."""

    return _FakeFit(
        variables={
            "population_loc_z": np.linspace(-0.2, 0.2, n_draws * n_params).reshape(n_draws, n_params),
            "population_log_scale": np.linspace(-1.1, -0.8, n_draws * n_params).reshape(n_draws, n_params),
            "subject_z": np.full((n_draws, n_subjects, n_params), 0.11, dtype=float),
            "subject_param": np.full((n_draws, n_subjects, n_params), 0.42, dtype=float),
            "block_z": np.full((n_draws, n_blocks, n_params), 0.11, dtype=float),
            "block_param": np.full((n_draws, n_blocks, n_params), 0.42, dtype=float),
            "log_likelihood_total": np.linspace(-10.0, -9.0, n_draws),
            "log_prior_total": np.linspace(-2.0, -1.0, n_draws),
            "log_posterior_total": np.linspace(-12.0, -10.0, n_draws),
        }
    )


def _fake_study_subject_block_fit(
    *,
    n_draws: int,
    n_subjects: int,
    n_blocks: int,
    n_params: int,
) -> _FakeFit:
    """Construct a fake CmdStan fit object for the population -> subject -> block case."""

    return _FakeFit(
        variables={
            "population_loc_z": np.linspace(-0.2, 0.2, n_draws * n_params).reshape(n_draws, n_params),
            "population_log_scale": np.linspace(-1.1, -0.8, n_draws * n_params).reshape(n_draws, n_params),
            "subject_loc_z": np.full((n_draws, n_subjects, n_params), 0.2, dtype=float),
            "subject_log_scale": np.full((n_draws, n_subjects, n_params), -0.9, dtype=float),
            "subject_param": np.full((n_draws, n_subjects, n_params), 0.55, dtype=float),
            "block_z": np.full((n_draws, n_blocks, n_params), 0.25, dtype=float),
            "block_param": np.full((n_draws, n_blocks, n_params), 0.61, dtype=float),
            "log_likelihood_total": np.linspace(-12.0, -11.0, n_draws),
            "log_prior_total": np.linspace(-3.0, -2.0, n_draws),
            "log_posterior_total": np.linspace(-15.0, -13.0, n_draws),
        }
    )


def test_draw_subject_block_hierarchy_posterior_stan_decodes_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Subject -> block Stan helper should decode backend draws."""

    subject = SubjectData(
        subject_id="s1",
        blocks=(
            BlockData(block_id="b1", trials=(_trial(0, 1, 1.0), _trial(1, 0, 0.0))),
            BlockData(block_id="b2", trials=(_trial(0, 0, 0.0), _trial(1, 1, 1.0))),
        ),
    )
    fake_fit = _fake_subject_block_fit(n_draws=6, n_blocks=2, n_params=1)

    monkeypatch.setattr(hierarchical_stan_module, "_run_stan_hierarchical_nuts", lambda **kwargs: fake_fit)

    result = draw_subject_block_hierarchy_posterior_stan(
        subject,
        model_component_id="asocial_state_q_value_softmax",
        model_kwargs={"beta": 2.0, "initial_value": 0.0},
        parameter_names=["alpha"],
        n_samples=3,
        n_warmup=2,
        thin=1,
        n_chains=2,
        random_seed=10,
    )

    assert result.subject_id == "s1"
    assert result.parameter_names == ("alpha",)
    assert len(result.draws) == 6
    assert result.draws[0].candidate.block_params[0]["alpha"] == pytest.approx(0.37)
    assert result.draws[0].candidate.subject_scale["alpha"] == pytest.approx(np.exp(-1.0))
    assert result.diagnostics.method == "subject_block_hierarchy_stan_nuts"


def test_estimate_subject_shared_map_stan_uses_shared_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Subject-shared Stan MAP helper should use the shared cache tag and init data."""

    subject = SubjectData(
        subject_id="s1",
        blocks=(
            BlockData(block_id="b1", trials=(_trial(0, 1, 1.0),)),
            BlockData(block_id="b2", trials=(_trial(0, 0, 0.0),)),
        ),
    )
    fake_fit = _fake_subject_shared_fit(n_draws=1, n_blocks=2, n_params=1)
    captured: dict[str, object] = {}

    def _fake_run(**kwargs: object) -> _FakeFit:
        captured.update(kwargs)
        return fake_fit

    monkeypatch.setattr(hierarchical_stan_module, "_run_stan_hierarchical_optimize", _fake_run)

    result = estimate_subject_shared_map_stan(
        subject,
        model_component_id="asocial_state_q_value_softmax",
        model_kwargs={"beta": 2.0, "initial_value": 0.0},
        parameter_names=["alpha"],
        method="lbfgs",
        max_iterations=50,
        random_seed=13,
    )

    assert result.subject_id == "s1"
    assert result.diagnostics.method == "subject_shared_stan_map"
    assert captured["cache_tag"] == "subject_shared_asocial_state_q_value_softmax"
    assert captured["init_data"] == {"subject_param_z": [0.0]}
    assert result.map_candidate.subject_params["alpha"] == pytest.approx(0.52)


def test_draw_study_subject_hierarchy_posterior_stan_decodes_population_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Population -> subject Stan helper should decode population and subject draws."""

    study = StudyData(
        subjects=(
            SubjectData(
                subject_id="s1",
                blocks=(
                    BlockData(block_id="b1", trials=(_trial(0, 1, 1.0),)),
                    BlockData(block_id="b2", trials=(_trial(0, 0, 0.0),)),
                ),
            ),
            SubjectData(
                subject_id="s2",
                blocks=(BlockData(block_id="b3", trials=(_trial(0, 1, 1.0),)),),
            ),
        )
    )
    fake_fit = _fake_study_subject_fit(n_draws=4, n_subjects=2, n_blocks=3, n_params=1)

    monkeypatch.setattr(hierarchical_stan_module, "_run_stan_hierarchical_nuts", lambda **kwargs: fake_fit)

    result = draw_study_subject_hierarchy_posterior_stan(
        study,
        model_component_id="asocial_state_q_value_softmax",
        model_kwargs={"beta": 2.0, "initial_value": 0.0},
        parameter_names=["alpha"],
        n_samples=2,
        n_warmup=1,
        thin=1,
        n_chains=2,
        random_seed=5,
    )

    assert result.subject_ids == ("s1", "s2")
    assert result.block_ids_by_subject == (("b1", "b2"), ("b3",))
    assert len(result.draws) == 4
    assert result.map_candidate.subject_params[0]["alpha"] == pytest.approx(0.42)
    assert result.map_candidate.block_params_by_subject[0][1]["alpha"] == pytest.approx(0.42)
    assert result.diagnostics.method == "study_subject_hierarchy_stan_nuts"


def test_estimate_study_subject_block_hierarchy_map_stan_decodes_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Population -> subject -> block Stan MAP helper should decode subject and block structure."""

    study = StudyData(
        subjects=(
            SubjectData(
                subject_id="s1",
                blocks=(
                    BlockData(block_id="b1", trials=(_trial(0, 1, 1.0),)),
                    BlockData(block_id="b2", trials=(_trial(0, 0, 0.0),)),
                ),
            ),
            SubjectData(
                subject_id="s2",
                blocks=(BlockData(block_id="b3", trials=(_trial(0, 1, 1.0),)),),
            ),
        )
    )
    fake_fit = _fake_study_subject_block_fit(n_draws=1, n_subjects=2, n_blocks=3, n_params=1)
    captured: dict[str, object] = {}

    def _fake_run(**kwargs: object) -> _FakeFit:
        captured.update(kwargs)
        return fake_fit

    monkeypatch.setattr(hierarchical_stan_module, "_run_stan_hierarchical_optimize", _fake_run)

    result = estimate_study_subject_block_hierarchy_map_stan(
        study,
        model_component_id="asocial_state_q_value_softmax",
        model_kwargs={"beta": 2.0, "initial_value": 0.0},
        parameter_names=["alpha"],
        method="lbfgs",
        max_iterations=20,
        random_seed=8,
    )

    assert result.subject_ids == ("s1", "s2")
    assert result.diagnostics.method == "study_subject_block_hierarchy_stan_map"
    assert captured["cache_tag"] == "study_subject_block_hierarchy_asocial_state_q_value_softmax"
    assert result.map_candidate.subject_params[0]["alpha"] == pytest.approx(0.55)
    assert result.map_candidate.block_params_by_subject[0][0]["alpha"] == pytest.approx(0.61)


@pytest.mark.parametrize(
    ("component_id", "parameter_names", "model_kwargs"),
    [
        ("asocial_q_value_softmax", ["alpha"], {"beta": 2.0, "initial_value": 0.0}),
        (
            "asocial_state_q_value_softmax_perseveration",
            ["alpha", "kappa"],
            {"beta": 2.0, "initial_value": 0.0},
        ),
        (
            "asocial_state_q_value_softmax_split_alpha",
            ["alpha_1", "alpha_2"],
            {"beta": 2.0, "initial_value": 0.0},
        ),
    ],
)
def test_draw_subject_block_hierarchy_posterior_stan_supports_additional_asocial_models(
    monkeypatch: pytest.MonkeyPatch,
    component_id: str,
    parameter_names: list[str],
    model_kwargs: dict[str, float],
) -> None:
    """Subject -> block Stan helper should support all asocial component IDs."""

    subject = SubjectData(
        subject_id="s1",
        blocks=(BlockData(block_id="b1", trials=(_trial(0, 1, 1.0), _trial(1, 0, 0.0))),),
    )
    fake_fit = _fake_subject_block_fit(n_draws=4, n_blocks=1, n_params=len(parameter_names))
    captured: dict[str, object] = {}

    def _fake_run(**kwargs: object) -> _FakeFit:
        captured.update(kwargs)
        return fake_fit

    monkeypatch.setattr(hierarchical_stan_module, "_run_stan_hierarchical_nuts", _fake_run)

    result = draw_subject_block_hierarchy_posterior_stan(
        subject,
        model_component_id=component_id,
        model_kwargs=model_kwargs,
        parameter_names=parameter_names,
        n_samples=2,
        n_warmup=1,
        thin=1,
        n_chains=2,
    )

    assert result.parameter_names == tuple(parameter_names)
    assert captured["cache_tag"] == f"subject_block_hierarchy_{component_id}"


def test_subject_block_hierarchy_uses_explicit_block_latents_not_condition_pooling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Subject -> block inputs should not build condition-level latent indices."""

    subject = SubjectData(
        subject_id="s1",
        blocks=(
            BlockData(block_id="b1", metadata={"condition": "A"}, trials=(_trial(0, 1, 1.0),)),
            BlockData(block_id="b2", metadata={"condition": "A"}, trials=(_trial(0, 0, 0.0),)),
        ),
    )
    fake_fit = _fake_subject_block_fit(n_draws=2, n_blocks=2, n_params=1)
    captured: dict[str, object] = {}

    def _fake_run(**kwargs: object) -> _FakeFit:
        captured.update(kwargs)
        return fake_fit

    monkeypatch.setattr(hierarchical_stan_module, "_run_stan_hierarchical_nuts", _fake_run)
    draw_subject_block_hierarchy_posterior_stan(
        subject,
        model_component_id="asocial_state_q_value_softmax",
        model_kwargs={"beta": 2.0, "initial_value": 0.0},
        parameter_names=["alpha"],
        n_samples=1,
        n_warmup=0,
        n_chains=2,
    )

    stan_data = captured["stan_data"]
    assert isinstance(stan_data, dict)
    assert "C" not in stan_data
    assert "condition_idx" not in stan_data


def test_draw_subject_block_hierarchy_posterior_stan_supports_social_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Subject -> block Stan helper should accept supported social model component IDs."""

    subject = SubjectData(subject_id="s1", blocks=(BlockData(block_id="b1", trials=_social_trials()),))
    fake_fit = _fake_subject_block_fit(n_draws=4, n_blocks=1, n_params=1)
    captured: dict[str, object] = {}

    def _fake_run(**kwargs: object) -> _FakeFit:
        captured.update(kwargs)
        return fake_fit

    monkeypatch.setattr(hierarchical_stan_module, "_run_stan_hierarchical_nuts", _fake_run)
    result = draw_subject_block_hierarchy_posterior_stan(
        subject,
        model_component_id="social_observed_outcome_q",
        model_kwargs={},
        parameter_names=["alpha_observed"],
        n_samples=2,
        n_warmup=1,
        n_chains=2,
    )

    assert result.parameter_names == ("alpha_observed",)
    assert captured["cache_tag"] == "subject_block_hierarchy_social_observed_outcome_q"
    stan_data = captured["stan_data"]
    assert isinstance(stan_data, dict)
    assert stan_data["actor_code"][0][0] == 2
    assert stan_data["actor_code"][0][1] == 1


def test_social_stan_loader_uses_structure_specific_files() -> None:
    """Social Stan loaders should resolve generic structure files for all components."""

    for component_id in social_supported_component_ids():
        source = load_social_stan_code(component_id, "subject_block_hierarchy")
        assert "social subject -> block Bayesian estimation" in source
        assert "data {" in source

