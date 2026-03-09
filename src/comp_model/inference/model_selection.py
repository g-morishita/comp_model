"""Model-comparison fitting helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from comp_model.analysis.information_criteria import aic, bic
from comp_model.core.data import BlockData, StudyData, SubjectData, TrialDecision
from comp_model.core.events import EpisodeTrace, EventPhase
from comp_model.plugins import PluginRegistry, build_default_registry

from .best_fit_summary import extract_best_fit_summary
from .likelihood import LikelihoodProgram
from .mle.fitting import MLEFitSpec, _build_trace_fit_function, coerce_episode_trace

SelectionCriterion = Literal["log_likelihood", "aic", "bic"]


@dataclass(frozen=True, slots=True)
class CandidateFitSpec:
    """One candidate model specification for model comparison."""

    name: str
    fit_function: Callable[[EpisodeTrace], Any]
    n_parameters: int | None = None
    fit_subject_function: Callable[[SubjectData], Any] | None = None
    fit_study_function: Callable[[StudyData], Any] | None = None


@dataclass(frozen=True, slots=True)
class RegistryCandidateFitSpec:
    """Registry-backed candidate definition."""

    name: str
    model_component_id: str
    fit_spec: MLEFitSpec
    model_kwargs: dict[str, Any] | None = None
    n_parameters: int | None = None


@dataclass(frozen=True, slots=True)
class CandidateComparison:
    """Fit summary for one candidate model."""

    candidate_name: str
    log_likelihood: float
    n_parameters: int
    aic: float
    bic: float
    score: float
    fit_result: Any


@dataclass(frozen=True, slots=True)
class ModelComparisonResult:
    """Model-comparison output."""

    criterion: SelectionCriterion
    n_observations: int
    comparisons: tuple[CandidateComparison, ...]
    selected_candidate_name: str


def compare_candidate_models(
    data: EpisodeTrace | BlockData | Sequence[TrialDecision],
    *,
    candidate_specs: Sequence[CandidateFitSpec],
    criterion: SelectionCriterion = "log_likelihood",
    n_observations: int | None = None,
) -> ModelComparisonResult:
    """Fit and compare candidate models on one dataset."""

    trace = coerce_episode_trace(data)
    if not candidate_specs:
        raise ValueError("candidate_specs must not be empty")
    _validate_criterion(criterion)

    inferred_observations = _count_decision_events(trace)
    n_obs = inferred_observations if n_observations is None else int(n_observations)
    if n_obs <= 0:
        raise ValueError("n_observations must be > 0")

    comparisons: list[CandidateComparison] = []
    for spec in candidate_specs:
        fit_result = spec.fit_function(trace)
        best = extract_best_fit_summary(fit_result)
        log_likelihood = float(best.log_likelihood)
        n_parameters = int(spec.n_parameters) if spec.n_parameters is not None else int(len(best.params))
        if n_parameters < 0:
            raise ValueError(f"n_parameters must be >= 0 for candidate {spec.name!r}")

        aic_value = aic(log_likelihood=log_likelihood, n_parameters=n_parameters)
        bic_value = bic(
            log_likelihood=log_likelihood,
            n_parameters=n_parameters,
            n_observations=n_obs,
        )
        comparisons.append(
            CandidateComparison(
                candidate_name=str(spec.name),
                log_likelihood=log_likelihood,
                n_parameters=n_parameters,
                aic=aic_value,
                bic=bic_value,
                score=_selection_score(
                    criterion=criterion,
                    log_likelihood=log_likelihood,
                    aic_value=aic_value,
                    bic_value=bic_value,
                ),
                fit_result=fit_result,
            )
        )

    selected = _select_candidate(comparisons, criterion=criterion)
    return ModelComparisonResult(
        criterion=criterion,
        n_observations=n_obs,
        comparisons=tuple(comparisons),
        selected_candidate_name=selected.candidate_name,
    )


def compare_registry_candidate_models(
    data: EpisodeTrace | BlockData | Sequence[TrialDecision],
    *,
    candidate_specs: Sequence[RegistryCandidateFitSpec],
    criterion: SelectionCriterion = "log_likelihood",
    n_observations: int | None = None,
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> ModelComparisonResult:
    """Fit and compare registry-backed model candidates."""

    reg = registry if registry is not None else build_default_registry()

    runtime_specs: list[CandidateFitSpec] = []
    for spec in candidate_specs:
        manifest = reg.get("model", spec.model_component_id)
        fixed_kwargs = dict(spec.model_kwargs) if spec.model_kwargs is not None else {}
        model_factory = lambda params, component_id=spec.model_component_id, fixed_kwargs=fixed_kwargs: reg.create_model(
            component_id,
            **_merge_kwargs(fixed_kwargs, params),
        )
        fit_function = _build_trace_fit_function(
            model_factory=model_factory,
            fit_spec=spec.fit_spec,
            requirements=manifest.requirements,
            likelihood_program=likelihood_program,
        )
        runtime_specs.append(
            CandidateFitSpec(
                name=spec.name,
                fit_function=fit_function,
                n_parameters=spec.n_parameters,
            )
        )

    return compare_candidate_models(
        data,
        candidate_specs=tuple(runtime_specs),
        criterion=criterion,
        n_observations=n_observations,
    )


def _count_decision_events(trace: EpisodeTrace) -> int:
    """Count decision observations in a canonical trace."""

    return sum(1 for event in trace.events if event.phase == EventPhase.DECISION)


def _validate_criterion(criterion: str) -> None:
    """Validate supported selection criterion."""

    if criterion not in {"log_likelihood", "aic", "bic"}:
        raise ValueError(
            "criterion must be one of "
            "{'log_likelihood', 'aic', 'bic'}"
        )


def _selection_score(
    *,
    criterion: SelectionCriterion,
    log_likelihood: float,
    aic_value: float,
    bic_value: float,
) -> float:
    """Compute criterion-specific score for one candidate."""

    if criterion == "log_likelihood":
        return float(log_likelihood)
    if criterion == "aic":
        return float(aic_value)
    if criterion == "bic":
        return float(bic_value)
    raise ValueError(f"unsupported criterion: {criterion!r}")


def _select_candidate(
    comparisons: Sequence[CandidateComparison],
    *,
    criterion: SelectionCriterion,
) -> CandidateComparison:
    """Select best candidate under the selected criterion."""

    if criterion == "log_likelihood":
        return max(comparisons, key=lambda item: item.score)
    return min(comparisons, key=lambda item: item.score)


def _merge_kwargs(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge fixed keyword arguments with free-parameter overrides."""

    merged = dict(base)
    merged.update(dict(override))
    return merged


__all__ = [
    "CandidateComparison",
    "CandidateFitSpec",
    "ModelComparisonResult",
    "RegistryCandidateFitSpec",
    "SelectionCriterion",
    "compare_candidate_models",
    "compare_registry_candidate_models",
]
