"""Explicit Stan posterior result dataclasses for hierarchical estimators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .compatibility import CompatibilityReport
from .mcmc_diagnostics import MCMCDiagnostics


def _best_candidate(draws: Sequence["StanPosteriorDraw"]) -> Any:
    """Return the retained candidate with the highest log posterior."""

    if not draws:
        raise ValueError("posterior result has no retained draws")
    return max(draws, key=lambda draw: draw.candidate.log_posterior).candidate


def _mean_params(param_rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Return per-parameter means across parameter rows."""

    if not param_rows:
        return {}
    out: dict[str, float] = {}
    names = tuple(str(name) for name in param_rows[0])
    for name in names:
        values = [float(row[name]) for row in param_rows if name in row]
        if values:
            out[name] = float(sum(values) / len(values))
    return out


def _flatten_nested_params(
    param_rows_by_subject: Sequence[Sequence[Mapping[str, float]]],
) -> list[Mapping[str, float]]:
    """Flatten nested block-parameter rows across subjects."""

    out: list[Mapping[str, float]] = []
    for subject_rows in param_rows_by_subject:
        out.extend(subject_rows)
    return out


@dataclass(frozen=True, slots=True)
class StanPosteriorDraw:
    """One retained Stan draw or optimized posterior point."""

    candidate: Any
    accepted: bool
    iteration: int


@dataclass(frozen=True, slots=True)
class SubjectSharedPosteriorCandidate:
    """One posterior candidate for a subject-shared parameter model."""

    parameter_names: tuple[str, ...]
    subject_params_z: dict[str, float]
    subject_params: dict[str, float]
    block_params_z: tuple[dict[str, float], ...]
    block_params: tuple[dict[str, float], ...]
    log_likelihood: float
    log_prior: float
    log_posterior: float


@dataclass(frozen=True, slots=True)
class SubjectSharedPosteriorResult:
    """Posterior output for one subject with parameters shared across blocks."""

    subject_id: str
    block_ids: tuple[str | int | None, ...]
    parameter_names: tuple[str, ...]
    draws: tuple[StanPosteriorDraw, ...]
    diagnostics: MCMCDiagnostics
    compatibility: CompatibilityReport | None = None

    @property
    def map_candidate(self) -> SubjectSharedPosteriorCandidate:
        """Return the highest-posterior retained candidate."""

        return _best_candidate(self.draws)

    @property
    def n_blocks(self) -> int:
        """Return the number of blocks represented in the fit."""

        return len(self.block_ids)

    @property
    def total_log_likelihood(self) -> float:
        """Return the MAP-candidate total log likelihood."""

        return float(self.map_candidate.log_likelihood)

    @property
    def total_log_prior(self) -> float:
        """Return the MAP-candidate total log prior."""

        return float(self.map_candidate.log_prior)

    @property
    def total_log_posterior(self) -> float:
        """Return the MAP-candidate total log posterior."""

        return float(self.map_candidate.log_posterior)

    @property
    def mean_map_params(self) -> dict[str, float]:
        """Return the shared MAP parameter mapping."""

        return dict(self.map_candidate.subject_params)


@dataclass(frozen=True, slots=True)
class SubjectBlockHierarchyPosteriorCandidate:
    """One posterior candidate for a subject -> block hierarchy."""

    parameter_names: tuple[str, ...]
    subject_location_z: dict[str, float]
    subject_scale: dict[str, float]
    block_params_z: tuple[dict[str, float], ...]
    block_params: tuple[dict[str, float], ...]
    log_likelihood: float
    log_prior: float
    log_posterior: float


@dataclass(frozen=True, slots=True)
class SubjectBlockHierarchyPosteriorResult:
    """Posterior output for one subject with block-specific parameters."""

    subject_id: str
    block_ids: tuple[str | int | None, ...]
    parameter_names: tuple[str, ...]
    draws: tuple[StanPosteriorDraw, ...]
    diagnostics: MCMCDiagnostics
    compatibility: CompatibilityReport | None = None

    @property
    def map_candidate(self) -> SubjectBlockHierarchyPosteriorCandidate:
        """Return the highest-posterior retained candidate."""

        return _best_candidate(self.draws)

    @property
    def n_blocks(self) -> int:
        """Return the number of blocks represented in the fit."""

        return len(self.block_ids)

    @property
    def total_log_likelihood(self) -> float:
        """Return the MAP-candidate total log likelihood."""

        return float(self.map_candidate.log_likelihood)

    @property
    def total_log_prior(self) -> float:
        """Return the MAP-candidate total log prior."""

        return float(self.map_candidate.log_prior)

    @property
    def total_log_posterior(self) -> float:
        """Return the MAP-candidate total log posterior."""

        return float(self.map_candidate.log_posterior)

    @property
    def mean_map_params(self) -> dict[str, float]:
        """Return mean MAP block parameters across blocks."""

        return _mean_params(self.map_candidate.block_params)


@dataclass(frozen=True, slots=True)
class StudySubjectHierarchyPosteriorCandidate:
    """One posterior candidate for a population -> subject hierarchy."""

    parameter_names: tuple[str, ...]
    population_location_z: dict[str, float]
    population_scale: dict[str, float]
    subject_params_z: tuple[dict[str, float], ...]
    subject_params: tuple[dict[str, float], ...]
    block_params_z_by_subject: tuple[tuple[dict[str, float], ...], ...]
    block_params_by_subject: tuple[tuple[dict[str, float], ...], ...]
    log_likelihood: float
    log_prior: float
    log_posterior: float


@dataclass(frozen=True, slots=True)
class StudySubjectHierarchyPosteriorResult:
    """Posterior output for a population -> subject hierarchy."""

    subject_ids: tuple[str, ...]
    block_ids_by_subject: tuple[tuple[str | int | None, ...], ...]
    parameter_names: tuple[str, ...]
    draws: tuple[StanPosteriorDraw, ...]
    diagnostics: MCMCDiagnostics
    compatibility_by_subject: tuple[CompatibilityReport | None, ...] | None = None

    @property
    def map_candidate(self) -> StudySubjectHierarchyPosteriorCandidate:
        """Return the highest-posterior retained candidate."""

        return _best_candidate(self.draws)

    @property
    def n_subjects(self) -> int:
        """Return the number of subjects represented in the fit."""

        return len(self.subject_ids)

    @property
    def total_log_likelihood(self) -> float:
        """Return the MAP-candidate total log likelihood."""

        return float(self.map_candidate.log_likelihood)

    @property
    def total_log_prior(self) -> float:
        """Return the MAP-candidate total log prior."""

        return float(self.map_candidate.log_prior)

    @property
    def total_log_posterior(self) -> float:
        """Return the MAP-candidate total log posterior."""

        return float(self.map_candidate.log_posterior)

    @property
    def mean_map_params_by_subject(self) -> dict[str, dict[str, float]]:
        """Return per-subject MAP parameter summaries."""

        return {
            subject_id: dict(subject_params)
            for subject_id, subject_params in zip(
                self.subject_ids,
                self.map_candidate.subject_params,
                strict=True,
            )
        }


@dataclass(frozen=True, slots=True)
class StudySubjectBlockHierarchyPosteriorCandidate:
    """One posterior candidate for a population -> subject -> block hierarchy."""

    parameter_names: tuple[str, ...]
    population_location_z: dict[str, float]
    population_scale: dict[str, float]
    subject_location_z: tuple[dict[str, float], ...]
    subject_scale: tuple[dict[str, float], ...]
    subject_params: tuple[dict[str, float], ...]
    block_params_z_by_subject: tuple[tuple[dict[str, float], ...], ...]
    block_params_by_subject: tuple[tuple[dict[str, float], ...], ...]
    log_likelihood: float
    log_prior: float
    log_posterior: float


@dataclass(frozen=True, slots=True)
class StudySubjectBlockHierarchyPosteriorResult:
    """Posterior output for a population -> subject -> block hierarchy."""

    subject_ids: tuple[str, ...]
    block_ids_by_subject: tuple[tuple[str | int | None, ...], ...]
    parameter_names: tuple[str, ...]
    draws: tuple[StanPosteriorDraw, ...]
    diagnostics: MCMCDiagnostics
    compatibility_by_subject: tuple[CompatibilityReport | None, ...] | None = None

    @property
    def map_candidate(self) -> StudySubjectBlockHierarchyPosteriorCandidate:
        """Return the highest-posterior retained candidate."""

        return _best_candidate(self.draws)

    @property
    def n_subjects(self) -> int:
        """Return the number of subjects represented in the fit."""

        return len(self.subject_ids)

    @property
    def total_log_likelihood(self) -> float:
        """Return the MAP-candidate total log likelihood."""

        return float(self.map_candidate.log_likelihood)

    @property
    def total_log_prior(self) -> float:
        """Return the MAP-candidate total log prior."""

        return float(self.map_candidate.log_prior)

    @property
    def total_log_posterior(self) -> float:
        """Return the MAP-candidate total log posterior."""

        return float(self.map_candidate.log_posterior)

    @property
    def mean_map_params_by_subject(self) -> dict[str, dict[str, float]]:
        """Return per-subject mean MAP block parameters."""

        return {
            subject_id: _mean_params(block_params)
            for subject_id, block_params in zip(
                self.subject_ids,
                self.map_candidate.block_params_by_subject,
                strict=True,
            )
        }


def mean_params_from_result(fit_result: Any) -> dict[str, float]:
    """Extract one representative parameter mapping from a hierarchical result."""

    if isinstance(fit_result, SubjectSharedPosteriorResult | SubjectBlockHierarchyPosteriorResult):
        return fit_result.mean_map_params
    if isinstance(fit_result, StudySubjectHierarchyPosteriorResult):
        return _mean_params(fit_result.map_candidate.subject_params)
    if isinstance(fit_result, StudySubjectBlockHierarchyPosteriorResult):
        return _mean_params(
            _flatten_nested_params(fit_result.map_candidate.block_params_by_subject)
        )
    raise TypeError(f"unsupported hierarchical posterior result: {type(fit_result)!r}")


__all__ = [
    "StanPosteriorDraw",
    "StudySubjectBlockHierarchyPosteriorCandidate",
    "StudySubjectBlockHierarchyPosteriorResult",
    "StudySubjectHierarchyPosteriorCandidate",
    "StudySubjectHierarchyPosteriorResult",
    "SubjectBlockHierarchyPosteriorCandidate",
    "SubjectBlockHierarchyPosteriorResult",
    "SubjectSharedPosteriorCandidate",
    "SubjectSharedPosteriorResult",
    "mean_params_from_result",
]
