"""Model-recovery workflow utilities.

This module compares candidate model families on synthetic datasets generated
from known models, then summarizes selection outcomes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import numpy as np

from comp_model.core.contracts import AgentModel, DecisionProblem
from comp_model.inference import MLEFitResult
from comp_model.runtime import SimulationConfig, run_episode

ObsT = TypeVar("ObsT")
ActionT = TypeVar("ActionT")
OutcomeT = TypeVar("OutcomeT")
SelectionCriterion = Literal["log_likelihood", "aic", "bic"]


@dataclass(frozen=True, slots=True)
class GeneratingModelSpec:
    """Generating model definition for model-recovery simulation.

    Parameters
    ----------
    name : str
        Label used for generating model in reports.
    model_factory : Callable[[dict[str, float]], AgentModel[ObsT, ActionT, OutcomeT]]
        Factory that builds a generating model from ``true_params``.
    true_params : dict[str, float]
        Ground-truth generating parameters.
    """

    name: str
    model_factory: Callable[[dict[str, float]], AgentModel[ObsT, ActionT, OutcomeT]]
    true_params: dict[str, float]


@dataclass(frozen=True, slots=True)
class CandidateModelSpec:
    """Candidate fitting model definition for model-recovery comparison.

    Parameters
    ----------
    name : str
        Candidate label used in reports.
    fit_function : Callable[[Any], MLEFitResult]
        Function fitting one trace and returning best log-likelihood.
    n_parameters : int | None, optional
        Number of effective free parameters for information criteria.
        If ``None``, this defaults to ``len(fit_result.best.params)``.
    """

    name: str
    fit_function: Callable[[Any], MLEFitResult]
    n_parameters: int | None = None


@dataclass(frozen=True, slots=True)
class CandidateFitSummary:
    """Per-candidate fit summary for one synthetic dataset.

    Parameters
    ----------
    candidate_name : str
        Candidate model label.
    log_likelihood : float
        Best log-likelihood from fitting.
    n_parameters : int
        Effective free parameter count.
    score : float
        Selection score under the chosen criterion.
    best_params : dict[str, float]
        Best-fit parameter mapping.
    """

    candidate_name: str
    log_likelihood: float
    n_parameters: int
    score: float
    best_params: dict[str, float]


@dataclass(frozen=True, slots=True)
class ModelRecoveryCase:
    """One generated dataset and candidate-model selection result.

    Parameters
    ----------
    case_index : int
        Zero-based dataset index.
    generating_model_name : str
        Name of the generating model used for this dataset.
    simulation_seed : int
        Seed used to generate this dataset.
    selected_candidate_name : str
        Candidate selected under the criterion.
    candidate_summaries : tuple[CandidateFitSummary, ...]
        Fit summaries for all candidates.
    """

    case_index: int
    generating_model_name: str
    simulation_seed: int
    selected_candidate_name: str
    candidate_summaries: tuple[CandidateFitSummary, ...]


@dataclass(frozen=True, slots=True)
class ModelRecoveryResult:
    """Output summary for model-recovery workflows.

    Parameters
    ----------
    cases : tuple[ModelRecoveryCase, ...]
        Per-dataset model-recovery records.
    confusion_matrix : dict[str, dict[str, int]]
        Nested counts ``confusion[generating][selected]``.
    criterion : {"log_likelihood", "aic", "bic"}
        Selection criterion used.
    """

    cases: tuple[ModelRecoveryCase, ...]
    confusion_matrix: dict[str, dict[str, int]]
    criterion: SelectionCriterion


def run_model_recovery(
    *,
    problem_factory: Callable[[], DecisionProblem[ObsT, ActionT, OutcomeT]],
    generating_specs: Sequence[GeneratingModelSpec],
    candidate_specs: Sequence[CandidateModelSpec],
    n_trials: int,
    n_replications_per_generator: int,
    criterion: SelectionCriterion = "log_likelihood",
    seed: int = 0,
) -> ModelRecoveryResult:
    """Run model-recovery simulation and candidate selection.

    Parameters
    ----------
    problem_factory : Callable[[], DecisionProblem[ObsT, ActionT, OutcomeT]]
        Factory returning a fresh problem instance.
    generating_specs : Sequence[GeneratingModelSpec]
        Generating model definitions.
    candidate_specs : Sequence[CandidateModelSpec]
        Candidate model fitting definitions.
    n_trials : int
        Number of trials per synthetic dataset.
    n_replications_per_generator : int
        Number of datasets to generate per generating model.
    criterion : {"log_likelihood", "aic", "bic"}, optional
        Selection criterion.
    seed : int, optional
        Master seed for deriving simulation seeds.

    Returns
    -------
    ModelRecoveryResult
        Case-level summaries and confusion matrix.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """

    if not generating_specs:
        raise ValueError("generating_specs must not be empty")
    if not candidate_specs:
        raise ValueError("candidate_specs must not be empty")
    if n_trials <= 0:
        raise ValueError("n_trials must be > 0")
    if n_replications_per_generator <= 0:
        raise ValueError("n_replications_per_generator must be > 0")

    rng = np.random.default_rng(seed)
    cases: list[ModelRecoveryCase] = []

    for generating in generating_specs:
        for _ in range(n_replications_per_generator):
            simulation_seed = int(rng.integers(0, 2**31 - 1))
            model = generating.model_factory(dict(generating.true_params))

            trace = run_episode(
                problem=problem_factory(),
                model=model,
                config=SimulationConfig(n_trials=n_trials, seed=simulation_seed),
            )

            candidate_summaries: list[CandidateFitSummary] = []
            for candidate in candidate_specs:
                fit_result = candidate.fit_function(trace)
                log_likelihood = float(fit_result.best.log_likelihood)
                n_parameters = (
                    int(candidate.n_parameters)
                    if candidate.n_parameters is not None
                    else int(len(fit_result.best.params))
                )
                score = _selection_score(
                    criterion=criterion,
                    log_likelihood=log_likelihood,
                    n_parameters=n_parameters,
                    n_observations=n_trials,
                )
                candidate_summaries.append(
                    CandidateFitSummary(
                        candidate_name=candidate.name,
                        log_likelihood=log_likelihood,
                        n_parameters=n_parameters,
                        score=score,
                        best_params={k: float(v) for k, v in fit_result.best.params.items()},
                    )
                )

            selected = _select_candidate(candidate_summaries, criterion=criterion)
            cases.append(
                ModelRecoveryCase(
                    case_index=len(cases),
                    generating_model_name=generating.name,
                    simulation_seed=simulation_seed,
                    selected_candidate_name=selected.candidate_name,
                    candidate_summaries=tuple(candidate_summaries),
                )
            )

    confusion = _build_confusion_matrix(cases)
    return ModelRecoveryResult(
        cases=tuple(cases),
        confusion_matrix=confusion,
        criterion=criterion,
    )


def _selection_score(
    *,
    criterion: SelectionCriterion,
    log_likelihood: float,
    n_parameters: int,
    n_observations: int,
) -> float:
    """Compute candidate selection score for one criterion."""

    if criterion == "log_likelihood":
        return float(log_likelihood)
    if criterion == "aic":
        return float(2.0 * n_parameters - 2.0 * log_likelihood)
    if criterion == "bic":
        return float(np.log(float(n_observations)) * n_parameters - 2.0 * log_likelihood)
    raise ValueError(f"unsupported criterion: {criterion!r}")


def _select_candidate(
    candidate_summaries: Sequence[CandidateFitSummary],
    *,
    criterion: SelectionCriterion,
) -> CandidateFitSummary:
    """Select best candidate for one generated dataset."""

    if criterion == "log_likelihood":
        return max(candidate_summaries, key=lambda item: item.score)
    return min(candidate_summaries, key=lambda item: item.score)


def _build_confusion_matrix(cases: Sequence[ModelRecoveryCase]) -> dict[str, dict[str, int]]:
    """Aggregate case-level selections into a confusion matrix."""

    matrix: dict[str, dict[str, int]] = {}
    for case in cases:
        by_selected = matrix.setdefault(case.generating_model_name, {})
        by_selected[case.selected_candidate_name] = by_selected.get(case.selected_candidate_name, 0) + 1
    return matrix


__all__ = [
    "CandidateFitSummary",
    "CandidateModelSpec",
    "GeneratingModelSpec",
    "ModelRecoveryCase",
    "ModelRecoveryResult",
    "run_model_recovery",
]
