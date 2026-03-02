"""Legacy hierarchical MAP API placeholders.

The previous SciPy-based hierarchical MAP implementation has been removed.
Bayesian estimation is Stan-only in this package.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from comp_model.core.data import StudyData, SubjectData
from comp_model.core.requirements import ComponentRequirements

from .likelihood import LikelihoodProgram
from .mle import ScipyMinimizeDiagnostics
from .transforms import ParameterTransform


@dataclass(frozen=True, slots=True)
class HierarchicalBlockResult:
    """MAP summary for one subject block."""

    block_id: str | int | None
    params: dict[str, float]
    log_likelihood: float


@dataclass(frozen=True, slots=True)
class HierarchicalSubjectMapResult:
    """Within-subject hierarchical MAP output."""

    subject_id: str
    parameter_names: tuple[str, ...]
    group_location_z: dict[str, float]
    group_scale_z: dict[str, float]
    block_results: tuple[HierarchicalBlockResult, ...]
    total_log_likelihood: float
    total_log_prior: float
    total_log_posterior: float
    scipy_diagnostics: ScipyMinimizeDiagnostics


@dataclass(frozen=True, slots=True)
class HierarchicalStudyMapResult:
    """Hierarchical MAP output aggregated across study subjects."""

    subject_results: tuple[HierarchicalSubjectMapResult, ...]
    total_log_likelihood: float
    total_log_prior: float
    total_log_posterior: float

    @property
    def n_subjects(self) -> int:
        """Return number of fitted subjects."""

        return len(self.subject_results)


def fit_subject_hierarchical_map(
    subject: SubjectData,
    *,
    model_factory: Any,
    parameter_names: Sequence[str],
    transforms: Mapping[str, ParameterTransform] | None = None,
    likelihood_program: LikelihoodProgram | None = None,
    requirements: ComponentRequirements | None = None,
    initial_group_location: Mapping[str, float] | None = None,
    initial_group_scale: Mapping[str, float] | None = None,
    initial_block_params: Sequence[Mapping[str, float]] | None = None,
    mu_prior_mean: float = 0.0,
    mu_prior_std: float = 2.0,
    log_sigma_prior_mean: float = -1.0,
    log_sigma_prior_std: float = 1.0,
    method: str = "L-BFGS-B",
    tol: float | None = None,
    options: Mapping[str, Any] | None = None,
) -> HierarchicalSubjectMapResult:
    """Removed SciPy hierarchical MAP API."""

    raise RuntimeError(
        "fit_subject_hierarchical_map has been removed. "
        "Use Stan estimators with estimator.type='within_subject_hierarchical_stan_map' "
        "or estimator.type='within_subject_hierarchical_stan_nuts'."
    )


def fit_study_hierarchical_map(
    study: StudyData,
    *,
    model_factory: Any,
    parameter_names: Sequence[str],
    transforms: Mapping[str, ParameterTransform] | None = None,
    likelihood_program: LikelihoodProgram | None = None,
    requirements: ComponentRequirements | None = None,
    initial_group_location: Mapping[str, float] | None = None,
    initial_group_scale: Mapping[str, float] | None = None,
    initial_block_params_by_subject: Mapping[str, Sequence[Mapping[str, float]]] | None = None,
    mu_prior_mean: float = 0.0,
    mu_prior_std: float = 2.0,
    log_sigma_prior_mean: float = -1.0,
    log_sigma_prior_std: float = 1.0,
    method: str = "L-BFGS-B",
    tol: float | None = None,
    options: Mapping[str, Any] | None = None,
) -> HierarchicalStudyMapResult:
    """Removed SciPy hierarchical MAP API."""

    raise RuntimeError(
        "fit_study_hierarchical_map has been removed. "
        "Use Stan estimators with estimator.type='within_subject_hierarchical_stan_map' "
        "or estimator.type='within_subject_hierarchical_stan_nuts'."
    )


__all__ = [
    "HierarchicalBlockResult",
    "HierarchicalStudyMapResult",
    "HierarchicalSubjectMapResult",
    "fit_study_hierarchical_map",
    "fit_subject_hierarchical_map",
]
