"""Model-comparison fitting helpers.

This module provides reusable utilities to fit multiple candidate models on
the same dataset and rank them under a selected criterion. Recovery workflows
reuse this exact path so user-facing fitting and recovery stay consistent.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from comp_model.analysis.information_criteria import aic, bic
from comp_model.core.data import BlockData, TrialDecision
from comp_model.core.events import EpisodeTrace, EventPhase
from comp_model.plugins import PluginRegistry, build_default_registry

from .fitting import FitSpec, build_model_fit_function, coerce_episode_trace
from .likelihood import LikelihoodProgram
from .mle import MLEFitResult

SelectionCriterion = Literal["log_likelihood", "aic", "bic"]


@dataclass(frozen=True, slots=True)
class CandidateFitSpec:
    """One candidate model specification for model comparison.

    Parameters
    ----------
    name : str
        Candidate label used in outputs.
    fit_function : Callable[[EpisodeTrace], MLEFitResult]
        Function fitting one canonical episode trace.
    n_parameters : int | None, optional
        Effective free parameter count used by information criteria.
        If ``None``, ``len(fit_result.best.params)`` is used.
    """

    name: str
    fit_function: Callable[[EpisodeTrace], MLEFitResult]
    n_parameters: int | None = None


@dataclass(frozen=True, slots=True)
class RegistryCandidateFitSpec:
    """Registry-backed candidate definition.

    Parameters
    ----------
    name : str
        Candidate label used in outputs.
    model_component_id : str
        Plugin model component ID.
    fit_spec : FitSpec
        Estimator specification for this candidate.
    model_kwargs : dict[str, Any] | None, optional
        Fixed keyword arguments passed to model construction.
    n_parameters : int | None, optional
        Effective free parameter count used by information criteria.
        If ``None``, ``len(fit_result.best.params)`` is used.
    """

    name: str
    model_component_id: str
    fit_spec: FitSpec
    model_kwargs: dict[str, Any] | None = None
    n_parameters: int | None = None


@dataclass(frozen=True, slots=True)
class CandidateComparison:
    """Fit summary for one candidate model.

    Parameters
    ----------
    candidate_name : str
        Candidate label.
    log_likelihood : float
        Best log-likelihood for this candidate.
    n_parameters : int
        Effective free parameter count.
    aic : float
        Akaike Information Criterion (lower is better).
    bic : float
        Bayesian Information Criterion (lower is better).
    score : float
        Criterion-specific selection score.
    fit_result : MLEFitResult
        Full fit output for downstream inspection.
    """

    candidate_name: str
    log_likelihood: float
    n_parameters: int
    aic: float
    bic: float
    score: float
    fit_result: MLEFitResult


@dataclass(frozen=True, slots=True)
class ModelComparisonResult:
    """Model-comparison output.

    Parameters
    ----------
    criterion : {"log_likelihood", "aic", "bic"}
        Selection criterion.
    n_observations : int
        Number of decision observations used for BIC computation.
    comparisons : tuple[CandidateComparison, ...]
        Candidate summaries in input order.
    selected_candidate_name : str
        Name of candidate selected under ``criterion``.
    """

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
    """Fit and compare candidate models on one dataset.

    Parameters
    ----------
    data : EpisodeTrace | BlockData | Sequence[TrialDecision]
        Dataset container supported by :func:`coerce_episode_trace`.
    candidate_specs : Sequence[CandidateFitSpec]
        Candidate fitting definitions.
    criterion : {"log_likelihood", "aic", "bic"}, optional
        Selection criterion.
    n_observations : int | None, optional
        Observation count for BIC. If ``None``, inferred as the number of
        decision events in the canonical trace.

    Returns
    -------
    ModelComparisonResult
        Candidate summaries and selected model label.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """

    trace = coerce_episode_trace(data)
    if not candidate_specs:
        raise ValueError("candidate_specs must not be empty")

    if criterion not in {"log_likelihood", "aic", "bic"}:
        raise ValueError("criterion must be one of {'log_likelihood', 'aic', 'bic'}")

    inferred_observations = _count_decision_events(trace)
    n_obs = inferred_observations if n_observations is None else int(n_observations)
    if n_obs <= 0:
        raise ValueError("n_observations must be > 0")

    comparisons: list[CandidateComparison] = []
    for spec in candidate_specs:
        fit_result = spec.fit_function(trace)
        log_likelihood = float(fit_result.best.log_likelihood)
        n_parameters = (
            int(spec.n_parameters)
            if spec.n_parameters is not None
            else int(len(fit_result.best.params))
        )
        if n_parameters < 0:
            raise ValueError(f"n_parameters must be >= 0 for candidate {spec.name!r}")

        aic_value = aic(log_likelihood=log_likelihood, n_parameters=n_parameters)
        bic_value = bic(
            log_likelihood=log_likelihood,
            n_parameters=n_parameters,
            n_observations=n_obs,
        )
        score = _selection_score(
            criterion=criterion,
            log_likelihood=log_likelihood,
            aic_value=aic_value,
            bic_value=bic_value,
        )

        comparisons.append(
            CandidateComparison(
                candidate_name=str(spec.name),
                log_likelihood=log_likelihood,
                n_parameters=n_parameters,
                aic=aic_value,
                bic=bic_value,
                score=score,
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
    """Fit and compare registry-backed model candidates.

    Parameters
    ----------
    data : EpisodeTrace | BlockData | Sequence[TrialDecision]
        Dataset container supported by :func:`coerce_episode_trace`.
    candidate_specs : Sequence[RegistryCandidateFitSpec]
        Registry-based candidate definitions.
    criterion : {"log_likelihood", "aic", "bic"}, optional
        Selection criterion.
    n_observations : int | None, optional
        Observation count for BIC. If ``None``, inferred from data.
    registry : PluginRegistry | None, optional
        Optional plugin registry instance.
    likelihood_program : LikelihoodProgram | None, optional
        Likelihood evaluator for candidate fitting.

    Returns
    -------
    ModelComparisonResult
        Candidate summaries and selected model label.
    """

    reg = registry if registry is not None else build_default_registry()

    runtime_specs: list[CandidateFitSpec] = []
    for spec in candidate_specs:
        manifest = reg.get("model", spec.model_component_id)
        fixed_kwargs = dict(spec.model_kwargs) if spec.model_kwargs is not None else {}
        model_factory = lambda params, component_id=spec.model_component_id, fixed_kwargs=fixed_kwargs: reg.create_model(
            component_id,
            **_merge_kwargs(fixed_kwargs, params),
        )
        fit_function = build_model_fit_function(
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
