"""Model-recovery workflow utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np

from comp_model.core.contracts import AgentModel, DecisionProblem
from comp_model.core.data import StudyData, SubjectData
from comp_model.inference.best_fit_summary import extract_best_fit_summary
from comp_model.inference.block_strategy import BlockFitStrategy, coerce_block_fit_strategy
from comp_model.inference.model_selection import (
    CandidateFitSpec,
    SelectionCriterion,
    compare_candidate_models,
)
from comp_model.inference.study_model_selection import (
    compare_study_candidate_models,
    compare_subject_candidate_models,
)
from comp_model.runtime import SimulationConfig, run_episode

ObsT = TypeVar("ObsT")
ActionT = TypeVar("ActionT")
OutcomeT = TypeVar("OutcomeT")


@dataclass(frozen=True, slots=True)
class GeneratingModelSpec:
    """Generating model definition for model-recovery simulation."""

    name: str
    model_factory: Callable[[dict[str, float]], AgentModel[ObsT, ActionT, OutcomeT]]
    true_params: dict[str, float]


@dataclass(frozen=True, slots=True)
class CandidateModelSpec:
    """Candidate fitting model definition for model-recovery comparison."""

    name: str
    fit_function: Callable[[Any], Any]
    n_parameters: int | None = None
    fit_subject_function: Callable[[SubjectData], Any] | None = None
    fit_study_function: Callable[[StudyData], Any] | None = None


@dataclass(frozen=True, slots=True)
class CandidateFitSummary:
    """Per-candidate fit summary for one synthetic dataset."""

    candidate_name: str
    log_likelihood: float
    n_parameters: int
    aic: float
    bic: float
    score: float
    best_params: dict[str, float]


@dataclass(frozen=True, slots=True)
class ModelRecoveryCase:
    """One generated dataset and candidate-model selection result."""

    case_index: int
    generating_model_name: str
    simulation_seed: int
    selected_candidate_name: str
    candidate_summaries: tuple[CandidateFitSummary, ...]


@dataclass(frozen=True, slots=True)
class ModelRecoveryResult:
    """Output summary for model-recovery workflows."""

    cases: tuple[ModelRecoveryCase, ...]
    confusion_matrix: dict[str, dict[str, int]]
    criterion: SelectionCriterion


def run_model_recovery(
    *,
    problem_factory: Callable[[], DecisionProblem[ObsT, ActionT, OutcomeT]] | None = None,
    generating_specs: Sequence[GeneratingModelSpec],
    candidate_specs: Sequence[CandidateModelSpec],
    n_trials: int,
    n_replications_per_generator: int,
    criterion: SelectionCriterion = "log_likelihood",
    seed: int = 0,
    trace_factory: Callable[[AgentModel[ObsT, ActionT, OutcomeT], int], Any] | None = None,
    block_fit_strategy: BlockFitStrategy = "independent",
) -> ModelRecoveryResult:
    """Run model-recovery simulation and candidate selection."""

    if not generating_specs:
        raise ValueError("generating_specs must not be empty")
    if not candidate_specs:
        raise ValueError("candidate_specs must not be empty")
    if n_trials <= 0:
        raise ValueError("n_trials must be > 0")
    if n_replications_per_generator <= 0:
        raise ValueError("n_replications_per_generator must be > 0")
    if trace_factory is None and problem_factory is None:
        raise ValueError("either problem_factory or trace_factory must be provided")
    strategy = coerce_block_fit_strategy(block_fit_strategy, field_name="block_fit_strategy")

    compiled_candidates = tuple(
        CandidateFitSpec(
            name=candidate.name,
            fit_function=candidate.fit_function,
            n_parameters=candidate.n_parameters,
            fit_subject_function=candidate.fit_subject_function,
            fit_study_function=candidate.fit_study_function,
        )
        for candidate in candidate_specs
    )

    rng = np.random.default_rng(seed)
    cases: list[ModelRecoveryCase] = []

    for generating in generating_specs:
        for _ in range(n_replications_per_generator):
            simulation_seed = int(rng.integers(0, 2**31 - 1))
            model: AgentModel[Any, Any, Any] = generating.model_factory(dict(generating.true_params))

            if trace_factory is not None:
                trace = trace_factory(model, simulation_seed)
            else:
                assert problem_factory is not None
                trace = run_episode(
                    problem=problem_factory(),
                    model=model,
                    config=SimulationConfig(n_trials=n_trials, seed=simulation_seed),
                )

            comparison: Any
            if isinstance(trace, SubjectData):
                comparison = compare_subject_candidate_models(
                    trace,
                    candidate_specs=compiled_candidates,
                    criterion=criterion,
                    block_fit_strategy=strategy,
                )
            elif isinstance(trace, StudyData):
                comparison = compare_study_candidate_models(
                    trace,
                    candidate_specs=compiled_candidates,
                    criterion=criterion,
                    block_fit_strategy=strategy,
                )
            else:
                comparison = compare_candidate_models(
                    trace,
                    candidate_specs=compiled_candidates,
                    criterion=criterion,
                )

            cases.append(
                ModelRecoveryCase(
                    case_index=len(cases),
                    generating_model_name=generating.name,
                    simulation_seed=simulation_seed,
                    selected_candidate_name=comparison.selected_candidate_name,
                    candidate_summaries=tuple(
                        _candidate_summary_from_comparison_item(item)
                        for item in comparison.comparisons
                    ),
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

    best = extract_best_fit_summary(item.fit_result) if hasattr(item, "fit_result") else None
    return CandidateFitSummary(
        candidate_name=item.candidate_name,
        log_likelihood=float(item.log_likelihood),
        n_parameters=int(item.n_parameters),
        aic=float(item.aic),
        bic=float(item.bic),
        score=float(item.score),
        best_params={key: float(value) for key, value in best.params.items()} if best is not None else {},
    )


__all__ = [
    "CandidateFitSummary",
    "CandidateModelSpec",
    "GeneratingModelSpec",
    "ModelRecoveryCase",
    "ModelRecoveryResult",
    "run_model_recovery",
]
