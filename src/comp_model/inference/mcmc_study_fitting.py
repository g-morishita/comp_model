"""Study-level MCMC posterior sampling workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from comp_model.core.data import BlockData, StudyData, SubjectData
from comp_model.plugins import PluginRegistry, build_default_registry

from .bayes import PriorProgram
from .mcmc import MCMCPosteriorResult, sample_posterior_model_from_registry


@dataclass(frozen=True, slots=True)
class MCMCBlockResult:
    """Posterior sampling output for one block.

    Parameters
    ----------
    block_id : str | int | None
        Block identifier.
    n_trials : int
        Number of trials in this block.
    posterior_result : MCMCPosteriorResult
        Posterior sampling output for this block.
    """

    block_id: str | int | None
    n_trials: int
    posterior_result: MCMCPosteriorResult


@dataclass(frozen=True, slots=True)
class MCMCSubjectResult:
    """Posterior sampling output for one subject across blocks.

    Parameters
    ----------
    subject_id : str
        Subject identifier.
    block_results : tuple[MCMCBlockResult, ...]
        Per-block posterior sampling outputs.
    total_map_log_likelihood : float
        Sum of block-level retained MAP log-likelihood values.
    total_map_log_posterior : float
        Sum of block-level retained MAP log-posterior values.
    mean_block_map_params : dict[str, float]
        Mean block-level retained MAP parameters across shared keys.
    """

    subject_id: str
    block_results: tuple[MCMCBlockResult, ...]
    total_map_log_likelihood: float
    total_map_log_posterior: float
    mean_block_map_params: dict[str, float]


@dataclass(frozen=True, slots=True)
class MCMCStudyResult:
    """Posterior sampling output for a full study.

    Parameters
    ----------
    subject_results : tuple[MCMCSubjectResult, ...]
        Per-subject posterior sampling summaries.
    total_map_log_likelihood : float
        Sum of subject-level retained MAP log-likelihood values.
    total_map_log_posterior : float
        Sum of subject-level retained MAP log-posterior values.
    """

    subject_results: tuple[MCMCSubjectResult, ...]
    total_map_log_likelihood: float
    total_map_log_posterior: float

    @property
    def n_subjects(self) -> int:
        """Return number of sampled subjects."""

        return len(self.subject_results)


def sample_posterior_block_data(
    block: BlockData,
    *,
    model_component_id: str,
    prior_program: PriorProgram,
    initial_params: Mapping[str, float],
    n_samples: int,
    n_warmup: int = 500,
    thin: int = 1,
    proposal_scales: Mapping[str, float] | None = None,
    bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
    registry: PluginRegistry | None = None,
    random_seed: int | None = None,
) -> MCMCBlockResult:
    """Sample posterior draws for one block dataset."""

    posterior_result = sample_posterior_model_from_registry(
        block,
        model_component_id=model_component_id,
        prior_program=prior_program,
        initial_params=initial_params,
        n_samples=n_samples,
        n_warmup=n_warmup,
        thin=thin,
        proposal_scales=proposal_scales,
        bounds=bounds,
        model_kwargs=model_kwargs,
        registry=registry,
        random_seed=random_seed,
    )
    return MCMCBlockResult(
        block_id=block.block_id,
        n_trials=block.n_trials,
        posterior_result=posterior_result,
    )


def sample_posterior_subject_data(
    subject: SubjectData,
    *,
    model_component_id: str,
    prior_program: PriorProgram,
    initial_params: Mapping[str, float],
    n_samples: int,
    n_warmup: int = 500,
    thin: int = 1,
    proposal_scales: Mapping[str, float] | None = None,
    bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
    registry: PluginRegistry | None = None,
    random_seed: int | None = None,
) -> MCMCSubjectResult:
    """Sample posterior draws independently for all blocks of one subject."""

    reg = registry if registry is not None else build_default_registry()
    block_results = tuple(
        sample_posterior_block_data(
            block,
            model_component_id=model_component_id,
            prior_program=prior_program,
            initial_params=initial_params,
            n_samples=n_samples,
            n_warmup=n_warmup,
            thin=thin,
            proposal_scales=proposal_scales,
            bounds=bounds,
            model_kwargs=model_kwargs,
            registry=reg,
            random_seed=None if random_seed is None else int(random_seed) + index,
        )
        for index, block in enumerate(subject.blocks)
    )

    total_map_log_likelihood = float(
        sum(
            block.posterior_result.map_candidate.log_likelihood
            for block in block_results
        )
    )
    total_map_log_posterior = float(
        sum(
            block.posterior_result.map_candidate.log_posterior
            for block in block_results
        )
    )
    mean_block_map_params = _mean_block_map_params(block_results)
    return MCMCSubjectResult(
        subject_id=subject.subject_id,
        block_results=block_results,
        total_map_log_likelihood=total_map_log_likelihood,
        total_map_log_posterior=total_map_log_posterior,
        mean_block_map_params=mean_block_map_params,
    )


def sample_posterior_study_data(
    study: StudyData,
    *,
    model_component_id: str,
    prior_program: PriorProgram,
    initial_params: Mapping[str, float],
    n_samples: int,
    n_warmup: int = 500,
    thin: int = 1,
    proposal_scales: Mapping[str, float] | None = None,
    bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
    registry: PluginRegistry | None = None,
    random_seed: int | None = None,
) -> MCMCStudyResult:
    """Sample posterior draws independently for all study subjects/blocks."""

    reg = registry if registry is not None else build_default_registry()
    subject_results = tuple(
        sample_posterior_subject_data(
            subject,
            model_component_id=model_component_id,
            prior_program=prior_program,
            initial_params=initial_params,
            n_samples=n_samples,
            n_warmup=n_warmup,
            thin=thin,
            proposal_scales=proposal_scales,
            bounds=bounds,
            model_kwargs=model_kwargs,
            registry=reg,
            random_seed=None if random_seed is None else int(random_seed) + index * 1000,
        )
        for index, subject in enumerate(study.subjects)
    )
    return MCMCStudyResult(
        subject_results=subject_results,
        total_map_log_likelihood=float(
            sum(item.total_map_log_likelihood for item in subject_results)
        ),
        total_map_log_posterior=float(
            sum(item.total_map_log_posterior for item in subject_results)
        ),
    )


def _mean_block_map_params(
    block_results: tuple[MCMCBlockResult, ...],
) -> dict[str, float]:
    """Average shared block-level retained MAP parameters."""

    if not block_results:
        return {}

    shared_keys: set[str] | None = None
    for block in block_results:
        keys = set(block.posterior_result.map_candidate.params)
        shared_keys = keys if shared_keys is None else (shared_keys & keys)

    if not shared_keys:
        return {}

    out: dict[str, float] = {}
    for key in sorted(shared_keys):
        values = [
            float(block.posterior_result.map_candidate.params[key])
            for block in block_results
        ]
        out[key] = float(sum(values) / len(values))
    return out


__all__ = [
    "MCMCBlockResult",
    "MCMCStudyResult",
    "MCMCSubjectResult",
    "sample_posterior_block_data",
    "sample_posterior_study_data",
    "sample_posterior_subject_data",
]
