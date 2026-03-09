"""Subject- and study-level model-comparison helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from comp_model.analysis.information_criteria import aic, bic
from comp_model.core.data import StudyData, SubjectData, get_block_trace
from comp_model.core.events import EventPhase

from .best_fit_summary import extract_best_fit_summary
from .block_strategy import BlockFitStrategy, coerce_block_fit_strategy
from .model_selection import CandidateFitSpec, SelectionCriterion


@dataclass(frozen=True, slots=True)
class SubjectCandidateComparison:
    """Candidate summary aggregated over all blocks for one subject."""

    candidate_name: str
    log_likelihood: float
    n_parameters: int
    aic: float
    bic: float
    score: float


@dataclass(frozen=True, slots=True)
class SubjectModelComparisonResult:
    """Model-comparison output for one subject."""

    subject_id: str
    criterion: SelectionCriterion
    n_observations: int
    comparisons: tuple[SubjectCandidateComparison, ...]
    selected_candidate_name: str


@dataclass(frozen=True, slots=True)
class StudyCandidateComparison:
    """Candidate summary aggregated over all study subjects."""

    candidate_name: str
    log_likelihood: float
    n_parameters: int
    aic: float
    bic: float
    score: float


@dataclass(frozen=True, slots=True)
class StudyModelComparisonResult:
    """Model-comparison output aggregated over study subjects."""

    criterion: SelectionCriterion
    n_observations: int
    subject_results: tuple[SubjectModelComparisonResult, ...]
    comparisons: tuple[StudyCandidateComparison, ...]
    selected_candidate_name: str

    @property
    def n_subjects(self) -> int:
        """Return number of compared subjects."""

        return len(self.subject_results)


def compare_subject_candidate_models(
    subject: SubjectData,
    *,
    candidate_specs: Sequence[CandidateFitSpec],
    criterion: SelectionCriterion = "log_likelihood",
    block_fit_strategy: BlockFitStrategy = "independent",
) -> SubjectModelComparisonResult:
    """Compare candidate models on all blocks for one subject."""

    if not candidate_specs:
        raise ValueError("candidate_specs must not be empty")
    _validate_criterion(criterion)
    strategy = coerce_block_fit_strategy(block_fit_strategy, field_name="block_fit_strategy")

    n_observations = 0
    block_traces = []
    for block in subject.blocks:
        trace = get_block_trace(block)
        block_traces.append(trace)
        n_observations += sum(1 for event in trace.events if event.phase == EventPhase.DECISION)
    if n_observations <= 0:
        raise ValueError("subject contains no decision observations")

    comparisons: list[SubjectCandidateComparison] = []
    for spec in candidate_specs:
        if strategy == "joint":
            comparison = _compare_candidate_joint_subject(
                subject=subject,
                spec=spec,
                criterion=criterion,
                n_observations=n_observations,
            )
        else:
            comparison = _compare_candidate_independent_subject(
                block_traces=tuple(block_traces),
                spec=spec,
                criterion=criterion,
                n_observations=n_observations,
            )
        comparisons.append(comparison)

    selected = _select_subject_candidate(comparisons, criterion=criterion)
    return SubjectModelComparisonResult(
        subject_id=subject.subject_id,
        criterion=criterion,
        n_observations=n_observations,
        comparisons=tuple(comparisons),
        selected_candidate_name=selected.candidate_name,
    )


def compare_study_candidate_models(
    study: StudyData,
    *,
    candidate_specs: Sequence[CandidateFitSpec],
    criterion: SelectionCriterion = "log_likelihood",
    block_fit_strategy: BlockFitStrategy = "independent",
) -> StudyModelComparisonResult:
    """Compare candidate models across all study subjects."""

    if not candidate_specs:
        raise ValueError("candidate_specs must not be empty")
    _validate_criterion(criterion)

    subject_results = tuple(
        compare_subject_candidate_models(
            subject,
            candidate_specs=candidate_specs,
            criterion=criterion,
            block_fit_strategy=block_fit_strategy,
        )
        for subject in study.subjects
    )
    total_observations = int(sum(item.n_observations for item in subject_results))
    if total_observations <= 0:
        raise ValueError("study contains no decision observations")

    by_name: dict[str, list[SubjectCandidateComparison]] = {}
    for subject_result in subject_results:
        for comparison in subject_result.comparisons:
            by_name.setdefault(comparison.candidate_name, []).append(comparison)

    comparisons: list[StudyCandidateComparison] = []
    for candidate_name, rows in by_name.items():
        total_log_likelihood = float(sum(item.log_likelihood for item in rows))
        n_parameters = int(rows[0].n_parameters)
        aic_value = aic(log_likelihood=total_log_likelihood, n_parameters=n_parameters)
        bic_value = bic(
            log_likelihood=total_log_likelihood,
            n_parameters=n_parameters,
            n_observations=total_observations,
        )
        comparisons.append(
            StudyCandidateComparison(
                candidate_name=str(candidate_name),
                log_likelihood=total_log_likelihood,
                n_parameters=n_parameters,
                aic=float(aic_value),
                bic=float(bic_value),
                score=float(
                    _selection_score(
                        criterion=criterion,
                        log_likelihood=total_log_likelihood,
                        aic_value=aic_value,
                        bic_value=bic_value,
                    )
                ),
            )
        )

    selected = _select_study_candidate(comparisons, criterion=criterion)
    return StudyModelComparisonResult(
        criterion=criterion,
        n_observations=total_observations,
        subject_results=subject_results,
        comparisons=tuple(comparisons),
        selected_candidate_name=selected.candidate_name,
    )


def _validate_criterion(criterion: str) -> None:
    """Validate supported selection criterion."""

    if criterion not in {"log_likelihood", "aic", "bic"}:
        raise ValueError(
            "criterion must be one of "
            "{'log_likelihood', 'aic', 'bic'}"
        )


def _compare_candidate_independent_subject(
    *,
    block_traces: tuple,
    spec: CandidateFitSpec,
    criterion: SelectionCriterion,
    n_observations: int,
) -> SubjectCandidateComparison:
    """Aggregate candidate fit metrics by fitting each block independently."""

    total_log_likelihood = 0.0
    first_param_count: int | None = None

    for trace in block_traces:
        fit_result = spec.fit_function(trace)
        best = extract_best_fit_summary(fit_result)
        total_log_likelihood += float(best.log_likelihood)
        if first_param_count is None:
            first_param_count = len(best.params)

    n_parameters = int(spec.n_parameters) if spec.n_parameters is not None else int(first_param_count or 0)
    if n_parameters < 0:
        raise ValueError(f"n_parameters must be >= 0 for candidate {spec.name!r}")

    aic_value = aic(log_likelihood=total_log_likelihood, n_parameters=n_parameters)
    bic_value = bic(
        log_likelihood=total_log_likelihood,
        n_parameters=n_parameters,
        n_observations=n_observations,
    )
    return SubjectCandidateComparison(
        candidate_name=str(spec.name),
        log_likelihood=float(total_log_likelihood),
        n_parameters=n_parameters,
        aic=float(aic_value),
        bic=float(bic_value),
        score=float(
            _selection_score(
                criterion=criterion,
                log_likelihood=total_log_likelihood,
                aic_value=aic_value,
                bic_value=bic_value,
            )
        ),
    )


def _compare_candidate_joint_subject(
    *,
    subject: SubjectData,
    spec: CandidateFitSpec,
    criterion: SelectionCriterion,
    n_observations: int,
) -> SubjectCandidateComparison:
    """Aggregate candidate fit metrics by fitting one shared subject model."""

    if spec.fit_subject_function is None:
        raise ValueError(
            f"candidate {spec.name!r} does not provide subject-level fitting "
            "required for block_fit_strategy='joint'"
        )

    fit_result = spec.fit_subject_function(subject)
    params, log_likelihood = _extract_joint_subject_summary(fit_result)
    n_parameters = int(spec.n_parameters) if spec.n_parameters is not None else int(len(params))
    if n_parameters < 0:
        raise ValueError(f"n_parameters must be >= 0 for candidate {spec.name!r}")

    aic_value = aic(log_likelihood=log_likelihood, n_parameters=n_parameters)
    bic_value = bic(
        log_likelihood=log_likelihood,
        n_parameters=n_parameters,
        n_observations=n_observations,
    )
    return SubjectCandidateComparison(
        candidate_name=str(spec.name),
        log_likelihood=float(log_likelihood),
        n_parameters=n_parameters,
        aic=float(aic_value),
        bic=float(bic_value),
        score=float(
            _selection_score(
                criterion=criterion,
                log_likelihood=log_likelihood,
                aic_value=aic_value,
                bic_value=bic_value,
            )
        ),
    )


def _extract_joint_subject_summary(fit_result: object) -> tuple[dict[str, float], float]:
    """Extract best summary for subject-level joint fit outputs."""

    total_log_likelihood = getattr(fit_result, "total_log_likelihood", None)
    if total_log_likelihood is not None:
        params_raw = getattr(fit_result, "shared_best_params", None)
        if isinstance(params_raw, dict):
            return (
                {str(key): float(value) for key, value in params_raw.items()},
                float(total_log_likelihood),
            )
        if getattr(fit_result, "fit_mode", None) == "independent":
            raise TypeError(
                "joint subject comparison requires a shared subject-level "
                "parameter estimate; got block-wise independent fits"
            )

    best = extract_best_fit_summary(fit_result)
    return ({str(key): float(value) for key, value in best.params.items()}, float(best.log_likelihood))


def _selection_score(
    *,
    criterion: SelectionCriterion,
    log_likelihood: float,
    aic_value: float,
    bic_value: float,
) -> float:
    """Compute criterion-specific score."""

    if criterion == "log_likelihood":
        return float(log_likelihood)
    if criterion == "aic":
        return float(aic_value)
    if criterion == "bic":
        return float(bic_value)
    raise ValueError(f"unsupported criterion: {criterion!r}")


def _select_subject_candidate(
    comparisons: Sequence[SubjectCandidateComparison],
    *,
    criterion: SelectionCriterion,
) -> SubjectCandidateComparison:
    """Select best subject-level candidate."""

    if criterion == "log_likelihood":
        return max(comparisons, key=lambda item: item.score)
    return min(comparisons, key=lambda item: item.score)


def _select_study_candidate(
    comparisons: Sequence[StudyCandidateComparison],
    *,
    criterion: SelectionCriterion,
) -> StudyCandidateComparison:
    """Select best study-level candidate."""

    if criterion == "log_likelihood":
        return max(comparisons, key=lambda item: item.score)
    return min(comparisons, key=lambda item: item.score)


__all__ = [
    "StudyCandidateComparison",
    "StudyModelComparisonResult",
    "SubjectCandidateComparison",
    "SubjectModelComparisonResult",
    "compare_study_candidate_models",
    "compare_subject_candidate_models",
]
