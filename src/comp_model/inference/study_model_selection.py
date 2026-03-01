"""Subject- and study-level model-comparison helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from comp_model.analysis.information_criteria import aic, bic
from comp_model.core.data import StudyData, SubjectData, get_block_trace
from comp_model.core.events import EventPhase

from .criteria import compute_pointwise_information_criteria
from .fit_result import extract_best_fit_summary
from .model_selection import CandidateFitSpec

SelectionCriterion = Literal["log_likelihood", "aic", "bic", "waic", "psis_loo"]


@dataclass(frozen=True, slots=True)
class SubjectCandidateComparison:
    """Candidate summary aggregated over all blocks for one subject.

    Parameters
    ----------
    candidate_name : str
        Candidate label.
    log_likelihood : float
        Total log-likelihood across subject blocks.
    log_posterior : float | None
        Total log-posterior across subject blocks when available.
    n_parameters : int
        Effective parameter count.
    aic : float
        AIC based on subject-level total log-likelihood.
    bic : float
        BIC based on subject-level total log-likelihood.
    score : float
        Criterion-specific selection score.
    waic : float | None, optional
        Sum of block-level WAIC values when available.
    psis_loo : float | None, optional
        Sum of block-level PSIS-LOO IC values when available.
    """

    candidate_name: str
    log_likelihood: float
    log_posterior: float | None
    n_parameters: int
    aic: float
    bic: float
    score: float
    waic: float | None = None
    psis_loo: float | None = None


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
    log_posterior: float | None
    n_parameters: int
    aic: float
    bic: float
    score: float
    waic: float | None = None
    psis_loo: float | None = None


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
) -> SubjectModelComparisonResult:
    """Compare candidate models on all blocks for one subject."""

    if not candidate_specs:
        raise ValueError("candidate_specs must not be empty")
    _validate_criterion(criterion)

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
        total_log_likelihood = 0.0
        total_log_posterior = 0.0
        has_posterior = True
        total_waic = 0.0
        total_looic = 0.0
        has_pointwise_criteria = True
        first_param_count: int | None = None

        for trace in block_traces:
            fit_result = spec.fit_function(trace)
            best = extract_best_fit_summary(fit_result)
            total_log_likelihood += float(best.log_likelihood)
            if best.log_posterior is None:
                has_posterior = False
            else:
                total_log_posterior += float(best.log_posterior)
            if first_param_count is None:
                first_param_count = len(best.params)
            try:
                block_waic, block_looic = compute_pointwise_information_criteria(fit_result)
                total_waic += float(block_waic)
                total_looic += float(block_looic)
            except (TypeError, ValueError) as exc:
                has_pointwise_criteria = False
                if criterion in {"waic", "psis_loo"}:
                    raise ValueError(
                        f"candidate {spec.name!r} does not support criterion "
                        f"{criterion!r}: {exc}"
                    ) from exc

        n_parameters = (
            int(spec.n_parameters)
            if spec.n_parameters is not None
            else int(first_param_count or 0)
        )
        if n_parameters < 0:
            raise ValueError(f"n_parameters must be >= 0 for candidate {spec.name!r}")

        aic_value = aic(log_likelihood=total_log_likelihood, n_parameters=n_parameters)
        bic_value = bic(
            log_likelihood=total_log_likelihood,
            n_parameters=n_parameters,
            n_observations=n_observations,
        )
        score = _selection_score(
            criterion=criterion,
            log_likelihood=total_log_likelihood,
            aic_value=aic_value,
            bic_value=bic_value,
            waic_value=(
                float(total_waic)
                if has_pointwise_criteria
                else None
            ),
            looic_value=(
                float(total_looic)
                if has_pointwise_criteria
                else None
            ),
        )
        comparisons.append(
            SubjectCandidateComparison(
                candidate_name=str(spec.name),
                log_likelihood=float(total_log_likelihood),
                log_posterior=(
                    float(total_log_posterior)
                    if has_posterior
                    else None
                ),
                n_parameters=n_parameters,
                aic=float(aic_value),
                bic=float(bic_value),
                waic=(
                    float(total_waic)
                    if has_pointwise_criteria
                    else None
                ),
                psis_loo=(
                    float(total_looic)
                    if has_pointwise_criteria
                    else None
                ),
                score=float(score),
            )
        )

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
        log_posterior_values = [item.log_posterior for item in rows]
        has_posterior = all(value is not None for value in log_posterior_values)
        total_log_posterior = (
            float(sum(float(value) for value in log_posterior_values if value is not None))
            if has_posterior
            else None
        )
        waic_values = [item.waic for item in rows]
        has_waic = all(value is not None for value in waic_values)
        total_waic = (
            float(sum(float(value) for value in waic_values if value is not None))
            if has_waic
            else None
        )
        looic_values = [item.psis_loo for item in rows]
        has_looic = all(value is not None for value in looic_values)
        total_looic = (
            float(sum(float(value) for value in looic_values if value is not None))
            if has_looic
            else None
        )

        # n_parameters is expected to be constant for one candidate.
        n_parameters = int(rows[0].n_parameters)
        aic_value = aic(log_likelihood=total_log_likelihood, n_parameters=n_parameters)
        bic_value = bic(
            log_likelihood=total_log_likelihood,
            n_parameters=n_parameters,
            n_observations=total_observations,
        )
        score = _selection_score(
            criterion=criterion,
            log_likelihood=total_log_likelihood,
            aic_value=aic_value,
            bic_value=bic_value,
            waic_value=total_waic,
            looic_value=total_looic,
        )
        comparisons.append(
            StudyCandidateComparison(
                candidate_name=str(candidate_name),
                log_likelihood=total_log_likelihood,
                log_posterior=total_log_posterior,
                n_parameters=n_parameters,
                aic=float(aic_value),
                bic=float(bic_value),
                waic=total_waic,
                psis_loo=total_looic,
                score=float(score),
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

    if criterion not in {"log_likelihood", "aic", "bic", "waic", "psis_loo"}:
        raise ValueError(
            "criterion must be one of "
            "{'log_likelihood', 'aic', 'bic', 'waic', 'psis_loo'}"
        )


def _selection_score(
    *,
    criterion: SelectionCriterion,
    log_likelihood: float,
    aic_value: float,
    bic_value: float,
    waic_value: float | None,
    looic_value: float | None,
) -> float:
    """Compute criterion-specific score."""

    if criterion == "log_likelihood":
        return float(log_likelihood)
    if criterion == "aic":
        return float(aic_value)
    if criterion == "bic":
        return float(bic_value)
    if criterion == "waic":
        if waic_value is None:
            raise ValueError("waic criterion requires pointwise posterior draws")
        return float(waic_value)
    if criterion == "psis_loo":
        if looic_value is None:
            raise ValueError("psis_loo criterion requires pointwise posterior draws")
        return float(looic_value)
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
