"""Tests for model-recovery workflow utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from comp_model.core.contracts import DecisionContext
from comp_model.demonstrators import FixedSequenceDemonstrator
from comp_model.inference import (
    ActionReplayLikelihood,
    ActorSubsetReplayLikelihood,
    BayesFitResult,
    FitSpec,
    GridSearchMLEEstimator,
    IndependentPriorProgram,
    MLECandidate,
    MLEFitResult,
    PosteriorCandidate,
    fit_model,
    sample_posterior_model,
    uniform_log_prior,
)
from comp_model.models import UniformRandomPolicyModel
from comp_model.problems import StationaryBanditProblem, TwoStageSocialBanditProgram
from comp_model.core.data import BlockData, StudyData, SubjectData, trial_decisions_from_trace
from comp_model.recovery import CandidateModelSpec, GeneratingModelSpec, run_model_recovery
from comp_model.runtime import SimulationConfig, run_episode, run_trial_program


@dataclass
class FixedChoiceModel:
    """Toy model with one free right-choice probability parameter."""

    p_right: float

    def start_episode(self) -> None:
        """No-op episode reset."""

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



def _fit_fixed_choice(trace: Any) -> MLEFitResult:
    """Fit fixed-choice model with grid-search MLE."""

    estimator = GridSearchMLEEstimator(
        likelihood_program=ActionReplayLikelihood(),
        model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
    )
    return estimator.fit(
        trace=trace,
        parameter_grid={"p_right": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
    )


def _fit_uniform_policy(trace: Any) -> MLEFitResult:
    """Compute likelihood under fixed uniform-random policy (no fitting)."""

    replay = ActionReplayLikelihood().evaluate(trace, UniformRandomPolicyModel())
    candidate = MLECandidate(params={}, log_likelihood=float(replay.total_log_likelihood))
    return MLEFitResult(best=candidate, candidates=(candidate,))


def _fit_constant_map(trace: Any) -> BayesFitResult:
    """Return a constant MAP fit result for compatibility tests."""

    candidate = PosteriorCandidate(
        params={"p_right": 0.8},
        log_likelihood=-5.0,
        log_prior=-0.5,
        log_posterior=-5.5,
    )
    return BayesFitResult(map_candidate=candidate, candidates=(candidate,))


def _fit_fixed_choice_mcmc(
    trace: Any,
    *,
    lower: float,
    upper: float,
    seed: int,
):
    """Fit fixed-choice model with MCMC posterior sampling."""

    return sample_posterior_model(
        trace,
        model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
        prior_program=IndependentPriorProgram(
            {"p_right": uniform_log_prior(lower=0.0, upper=1.0)}
        ),
        initial_params={"p_right": min(max(0.5, lower + 1e-3), upper - 1e-3)},
        n_samples=30,
        n_warmup=30,
        thin=1,
        proposal_scales={"p_right": 0.08},
        bounds={"p_right": (lower, upper)},
        random_seed=seed,
    )


def test_run_model_recovery_prefers_matching_candidate() -> None:
    """Model recovery should usually select generating-equivalent candidate."""

    result = run_model_recovery(
        problem_factory=lambda: StationaryBanditProblem([0.5, 0.5]),
        generating_specs=(
            GeneratingModelSpec(
                name="fixed_choice",
                model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
                true_params={"p_right": 0.85},
            ),
        ),
        candidate_specs=(
            CandidateModelSpec(name="fixed_choice_mle", fit_function=_fit_fixed_choice, n_parameters=1),
            CandidateModelSpec(name="uniform_random", fit_function=_fit_uniform_policy, n_parameters=0),
        ),
        n_trials=120,
        n_replications_per_generator=6,
        criterion="log_likelihood",
        seed=5,
    )

    assert len(result.cases) == 6
    fixed_hits = result.confusion_matrix["fixed_choice"].get("fixed_choice_mle", 0)
    assert fixed_hits >= 5


def test_run_model_recovery_supports_aic_selection() -> None:
    """Model recovery should support information-criterion based selection."""

    result = run_model_recovery(
        problem_factory=lambda: StationaryBanditProblem([0.5, 0.5]),
        generating_specs=(
            GeneratingModelSpec(
                name="fixed_choice",
                model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
                true_params={"p_right": 0.8},
            ),
        ),
        candidate_specs=(
            CandidateModelSpec(name="fixed_choice_mle", fit_function=_fit_fixed_choice, n_parameters=1),
            CandidateModelSpec(name="uniform_random", fit_function=_fit_uniform_policy, n_parameters=0),
        ),
        n_trials=120,
        n_replications_per_generator=4,
        criterion="aic",
        seed=8,
    )

    assert result.criterion == "aic"
    assert result.confusion_matrix["fixed_choice"].get("fixed_choice_mle", 0) >= 3


def test_run_model_recovery_validates_inputs() -> None:
    """Model recovery API should reject invalid configuration values."""

    with pytest.raises(ValueError, match="generating_specs must not be empty"):
        run_model_recovery(
            problem_factory=lambda: StationaryBanditProblem([0.5, 0.5]),
            generating_specs=(),
            candidate_specs=(CandidateModelSpec(name="fixed", fit_function=_fit_fixed_choice),),
            n_trials=20,
            n_replications_per_generator=1,
        )

    with pytest.raises(ValueError, match="candidate_specs must not be empty"):
        run_model_recovery(
            problem_factory=lambda: StationaryBanditProblem([0.5, 0.5]),
            generating_specs=(
                GeneratingModelSpec(
                    name="fixed_choice",
                    model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
                    true_params={"p_right": 0.8},
                ),
            ),
            candidate_specs=(),
            n_trials=20,
            n_replications_per_generator=1,
        )


def test_run_model_recovery_accepts_map_fit_results() -> None:
    """Model recovery should accept MAP-style candidate fit outputs."""

    result = run_model_recovery(
        problem_factory=lambda: StationaryBanditProblem([0.5, 0.5]),
        generating_specs=(
            GeneratingModelSpec(
                name="fixed_choice",
                model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
                true_params={"p_right": 0.8},
            ),
        ),
        candidate_specs=(
            CandidateModelSpec(name="map_candidate", fit_function=_fit_constant_map, n_parameters=1),
            CandidateModelSpec(name="uniform_random", fit_function=_fit_uniform_policy, n_parameters=0),
        ),
        n_trials=50,
        n_replications_per_generator=2,
        criterion="log_likelihood",
        seed=12,
    )

    assert len(result.cases) == 2
    assert result.confusion_matrix["fixed_choice"].get("map_candidate", 0) == 2
    for case in result.cases:
        map_summary = next(item for item in case.candidate_summaries if item.candidate_name == "map_candidate")
        assert map_summary.log_posterior == pytest.approx(-5.5)


def test_run_model_recovery_supports_waic_criterion_with_mcmc_candidates() -> None:
    """Model recovery should support WAIC criterion for posterior-fit candidates."""

    result = run_model_recovery(
        problem_factory=lambda: StationaryBanditProblem([0.5, 0.5]),
        generating_specs=(
            GeneratingModelSpec(
                name="fixed_choice",
                model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
                true_params={"p_right": 0.8},
            ),
        ),
        candidate_specs=(
            CandidateModelSpec(
                name="good_mcmc",
                fit_function=lambda trace: _fit_fixed_choice_mcmc(
                    trace, lower=0.0, upper=1.0, seed=5
                ),
                n_parameters=1,
            ),
            CandidateModelSpec(
                name="bad_mcmc",
                fit_function=lambda trace: _fit_fixed_choice_mcmc(
                    trace, lower=0.0, upper=0.4, seed=6
                ),
                n_parameters=1,
            ),
        ),
        n_trials=60,
        n_replications_per_generator=2,
        criterion="waic",
        seed=21,
    )

    assert result.criterion == "waic"
    assert len(result.cases) == 2
    assert result.confusion_matrix["fixed_choice"].get("good_mcmc", 0) == 2
    for case in result.cases:
        summary = next(item for item in case.candidate_summaries if item.candidate_name == "good_mcmc")
        assert summary.waic is not None
        assert summary.psis_loo is not None


def test_run_model_recovery_supports_custom_social_trace_factory() -> None:
    """Model recovery should support multi-actor traces via trace_factory hook."""

    def fit_fixed_choice_social(trace: Any) -> MLEFitResult:
        return fit_model(
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

    def fit_uniform_social(trace: Any) -> MLEFitResult:
        replay = ActorSubsetReplayLikelihood(
            fitted_actor_id="subject",
            scored_actor_ids=("subject",),
            auto_fill_unmodeled_actors=True,
        ).evaluate(trace, UniformRandomPolicyModel())
        candidate = MLECandidate(
            params={},
            log_likelihood=float(replay.total_log_likelihood),
        )
        return MLEFitResult(best=candidate, candidates=(candidate,))

    def social_trace_factory(model: Any, simulation_seed: int):
        return run_trial_program(
            program=TwoStageSocialBanditProgram([0.5, 0.5]),
            models={
                "subject": model,
                "demonstrator": FixedSequenceDemonstrator(sequence=[1] * 90),
            },
            config=SimulationConfig(n_trials=90, seed=simulation_seed),
        )

    result = run_model_recovery(
        problem_factory=None,
        generating_specs=(
            GeneratingModelSpec(
                name="fixed_choice_social",
                model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
                true_params={"p_right": 0.8},
            ),
        ),
        candidate_specs=(
            CandidateModelSpec(
                name="fixed_choice_social_fit",
                fit_function=fit_fixed_choice_social,
                n_parameters=1,
            ),
            CandidateModelSpec(
                name="uniform_social",
                fit_function=fit_uniform_social,
                n_parameters=0,
            ),
        ),
        n_trials=90,
        n_replications_per_generator=4,
        criterion="log_likelihood",
        seed=28,
        trace_factory=social_trace_factory,
    )

    assert len(result.cases) == 4
    assert result.confusion_matrix["fixed_choice_social"].get("fixed_choice_social_fit", 0) >= 3


def test_run_model_recovery_requires_problem_or_trace_factory() -> None:
    """Model recovery should reject missing simulation source definitions."""

    with pytest.raises(ValueError, match="either problem_factory or trace_factory must be provided"):
        run_model_recovery(
            problem_factory=None,
            generating_specs=(
                GeneratingModelSpec(
                    name="fixed_choice",
                    model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
                    true_params={"p_right": 0.8},
                ),
            ),
            candidate_specs=(
                CandidateModelSpec(name="fixed_choice_mle", fit_function=_fit_fixed_choice, n_parameters=1),
            ),
            n_trials=20,
            n_replications_per_generator=1,
        )


def test_run_model_recovery_supports_subject_level_datasets_from_trace_factory() -> None:
    """Recovery should compare candidates when trace_factory returns SubjectData."""

    def subject_trace_factory(model: Any, simulation_seed: int):
        block_traces = []
        for offset in (0, 1000):
            block_traces.append(
                run_episode(
                    problem=StationaryBanditProblem([0.5, 0.5]),
                    model=model,
                    config=SimulationConfig(n_trials=40, seed=simulation_seed + offset),
                )
            )
        blocks = tuple(
            BlockData(
                block_id=f"b{index}",
                trials=trial_decisions_from_trace(trace),
                event_trace=trace,
            )
            for index, trace in enumerate(block_traces)
        )
        return SubjectData(subject_id="s1", blocks=blocks)

    result = run_model_recovery(
        problem_factory=None,
        generating_specs=(
            GeneratingModelSpec(
                name="fixed_choice_subject",
                model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
                true_params={"p_right": 0.85},
            ),
        ),
        candidate_specs=(
            CandidateModelSpec(name="fixed_choice_mle", fit_function=_fit_fixed_choice, n_parameters=1),
            CandidateModelSpec(name="uniform_random", fit_function=_fit_uniform_policy, n_parameters=0),
        ),
        n_trials=40,
        n_replications_per_generator=2,
        criterion="log_likelihood",
        seed=61,
        trace_factory=subject_trace_factory,
    )

    assert len(result.cases) == 2
    assert result.confusion_matrix["fixed_choice_subject"].get("fixed_choice_mle", 0) >= 1


def test_run_model_recovery_supports_study_level_datasets_from_trace_factory() -> None:
    """Recovery should compare candidates when trace_factory returns StudyData."""

    def study_trace_factory(model: Any, simulation_seed: int):
        subject_rows: list[SubjectData] = []
        for subject_index in range(2):
            block_rows: list[BlockData] = []
            for block_index in range(2):
                trace = run_episode(
                    problem=StationaryBanditProblem([0.5, 0.5]),
                    model=model,
                    config=SimulationConfig(
                        n_trials=30,
                        seed=simulation_seed + subject_index * 1000 + block_index * 100,
                    ),
                )
                block_rows.append(
                    BlockData(
                        block_id=f"b{block_index}",
                        trials=trial_decisions_from_trace(trace),
                        event_trace=trace,
                    )
                )
            subject_rows.append(
                SubjectData(
                    subject_id=f"s{subject_index}",
                    blocks=tuple(block_rows),
                )
            )
        return StudyData(subjects=tuple(subject_rows))

    result = run_model_recovery(
        problem_factory=None,
        generating_specs=(
            GeneratingModelSpec(
                name="fixed_choice_study",
                model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
                true_params={"p_right": 0.85},
            ),
        ),
        candidate_specs=(
            CandidateModelSpec(name="fixed_choice_mle", fit_function=_fit_fixed_choice, n_parameters=1),
            CandidateModelSpec(name="uniform_random", fit_function=_fit_uniform_policy, n_parameters=0),
        ),
        n_trials=30,
        n_replications_per_generator=1,
        criterion="log_likelihood",
        seed=62,
        trace_factory=study_trace_factory,
    )

    assert len(result.cases) == 1
    case = result.cases[0]
    assert case.selected_candidate_name in {"fixed_choice_mle", "uniform_random"}
    for summary in case.candidate_summaries:
        assert isinstance(summary.best_params, dict)
