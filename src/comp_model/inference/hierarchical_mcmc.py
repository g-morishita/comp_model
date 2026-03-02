"""Shared hierarchical posterior result dataclasses.

This module contains only posterior result containers used by Stan-backed
hierarchical sampling.
"""

from __future__ import annotations

from dataclasses import dataclass

from .compatibility import CompatibilityReport
from .mcmc import MCMCDiagnostics


@dataclass(frozen=True, slots=True)
class HierarchicalPosteriorCandidate:
    """One evaluated hierarchical posterior candidate."""

    parameter_names: tuple[str, ...]
    group_location_z: dict[str, float]
    group_scale_z: dict[str, float]
    block_params_z: tuple[dict[str, float], ...]
    block_params: tuple[dict[str, float], ...]
    log_likelihood: float
    log_prior: float
    log_posterior: float


@dataclass(frozen=True, slots=True)
class HierarchicalMCMCDraw:
    """One retained hierarchical posterior draw."""

    candidate: HierarchicalPosteriorCandidate
    accepted: bool
    iteration: int


@dataclass(frozen=True, slots=True)
class HierarchicalSubjectPosteriorResult:
    """Within-subject hierarchical posterior sampling output."""

    subject_id: str
    parameter_names: tuple[str, ...]
    draws: tuple[HierarchicalMCMCDraw, ...]
    diagnostics: MCMCDiagnostics
    compatibility: CompatibilityReport | None = None

    @property
    def map_candidate(self) -> HierarchicalPosteriorCandidate:
        """Return highest-posterior retained candidate."""

        return max(self.draws, key=lambda draw: draw.candidate.log_posterior).candidate

    @property
    def n_blocks(self) -> int:
        """Return number of blocks represented in each draw."""

        if not self.draws:
            return 0
        return len(self.draws[0].candidate.block_params)


@dataclass(frozen=True, slots=True)
class HierarchicalStudyPosteriorResult:
    """Study-level hierarchical posterior sampling output."""

    subject_results: tuple[HierarchicalSubjectPosteriorResult, ...]

    @property
    def n_subjects(self) -> int:
        """Return number of sampled subjects."""

        return len(self.subject_results)

    @property
    def total_map_log_likelihood(self) -> float:
        """Return sum of subject-level MAP draw log-likelihood values."""

        return float(sum(item.map_candidate.log_likelihood for item in self.subject_results))

    @property
    def total_map_log_prior(self) -> float:
        """Return sum of subject-level MAP draw log-prior values."""

        return float(sum(item.map_candidate.log_prior for item in self.subject_results))

    @property
    def total_map_log_posterior(self) -> float:
        """Return sum of subject-level MAP draw log-posterior values."""

        return float(sum(item.map_candidate.log_posterior for item in self.subject_results))


__all__ = [
    "HierarchicalMCMCDraw",
    "HierarchicalPosteriorCandidate",
    "HierarchicalStudyPosteriorResult",
    "HierarchicalSubjectPosteriorResult",
]
