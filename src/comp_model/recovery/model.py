"""Model-recovery workflow utilities.

This module compares candidate model families on synthetic datasets generated
from known models, then summarizes selection outcomes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np

from comp_model.core.contracts import AgentModel, DecisionProblem
from comp_model.inference.fit_result import extract_best_fit_summary
from comp_model.inference.model_selection import (
    CandidateFitSpec,
    SelectionCriterion,
    compare_candidate_models,
)
from comp_model.runtime import SimulationConfig, run_episode

ObsT = TypeVar("ObsT")
ActionT = TypeVar("ActionT")
OutcomeT = TypeVar("OutcomeT")


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
    fit_function : Callable[[Any], Any]
        Function fitting one trace and returning a supported inference fit
        result (MLE-style or MAP-style).
    n_parameters : int | None, optional
        Number of effective free parameters for information criteria.
        If ``None``, this defaults to ``len(fit_result.best.params)``.
    """

    name: str
    fit_function: Callable[[Any], Any]
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

    # Compile one reusable candidate set so every simulated dataset is evaluated
    # with exactly the same fitting configuration.
    compiled_candidates = tuple(
        CandidateFitSpec(
            name=candidate.name,
            fit_function=candidate.fit_function,
            n_parameters=candidate.n_parameters,
        )
        for candidate in candidate_specs
    )

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

            comparison = compare_candidate_models(
                trace,
                candidate_specs=compiled_candidates,
                criterion=criterion,
            )

            candidate_summaries = tuple(
                _candidate_summary_from_comparison_item(item)
                for item in comparison.comparisons
            )

            cases.append(
                ModelRecoveryCase(
                    case_index=len(cases),
                    generating_model_name=generating.name,
                    simulation_seed=simulation_seed,
                    selected_candidate_name=comparison.selected_candidate_name,
                    candidate_summaries=candidate_summaries,
                )
            )

    confusion = _build_confusion_matrix(cases)
    return ModelRecoveryResult(
        cases=tuple(cases),
        confusion_matrix=confusion,
        criterion=criterion,
    )


def _build_confusion_matrix(cases: Sequence[ModelRecoveryCase]) -> dict[str, dict[str, int]]:
    """Aggregate case-level selections into a confusion matrix."""

    matrix: dict[str, dict[str, int]] = {}
    for case in cases:
        by_selected = matrix.setdefault(case.generating_model_name, {})
        by_selected[case.selected_candidate_name] = by_selected.get(case.selected_candidate_name, 0) + 1
    return matrix


def _candidate_summary_from_comparison_item(item: Any) -> CandidateFitSummary:
    """Convert one model-comparison record into recovery summary form."""

    best = extract_best_fit_summary(item.fit_result)
    return CandidateFitSummary(
        candidate_name=item.candidate_name,
        log_likelihood=float(item.log_likelihood),
        n_parameters=int(item.n_parameters),
        score=float(item.score),
        best_params={key: float(value) for key, value in best.params.items()},
    )


__all__ = [
    "CandidateFitSummary",
    "CandidateModelSpec",
    "GeneratingModelSpec",
    "ModelRecoveryCase",
    "ModelRecoveryResult",
    "run_model_recovery",
]
