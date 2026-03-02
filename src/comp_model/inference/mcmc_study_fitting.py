"""Study-level MCMC posterior sampling workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from comp_model.core.data import BlockData, StudyData, SubjectData, get_block_trace
from comp_model.plugins import PluginRegistry, build_default_registry

from .bayes import PriorProgram
from .block_strategy import (
    JOINT_BLOCK_ID,
    BlockFitStrategy,
    JointBlockLikelihoodProgram,
    coerce_block_fit_strategy,
)
from .compatibility import assert_trace_compatible
from .likelihood import LikelihoodProgram
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
    fit_mode : {"independent", "joint"}
        Block-fitting strategy used for this subject.
    input_n_blocks : int | None
        Number of blocks in the original input subject data.
    """

    subject_id: str
    block_results: tuple[MCMCBlockResult, ...]
    total_map_log_likelihood: float
    total_map_log_posterior: float
    mean_block_map_params: dict[str, float]
    fit_mode: BlockFitStrategy = "independent"
    input_n_blocks: int | None = None


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
    likelihood_program: LikelihoodProgram | None = None,
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
        likelihood_program=likelihood_program,
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
    likelihood_program: LikelihoodProgram | None = None,
    random_seed: int | None = None,
    block_fit_strategy: BlockFitStrategy = "independent",
) -> MCMCSubjectResult:
    """Sample posterior draws across all blocks of one subject.

    Parameters
    ----------
    subject : SubjectData
        Subject dataset.
    model_component_id : str
        Registered model component ID.
    prior_program : PriorProgram
        Prior evaluator.
    initial_params : Mapping[str, float]
        Initial constrained parameter values.
    n_samples : int
        Number of retained posterior draws.
    n_warmup : int, optional
        Warmup iterations.
    thin : int, optional
        Thinning interval.
    proposal_scales : Mapping[str, float] | None, optional
        Optional proposal scales.
    bounds : Mapping[str, tuple[float | None, float | None]] | None, optional
        Optional hard parameter bounds.
    model_kwargs : Mapping[str, Any] | None, optional
        Fixed model constructor kwargs.
    registry : PluginRegistry | None, optional
        Optional registry instance.
    likelihood_program : LikelihoodProgram | None, optional
        Optional likelihood evaluator.
    random_seed : int | None, optional
        Random seed.
    block_fit_strategy : {"independent", "joint"}, optional
        ``"independent"`` samples each block separately and aggregates block MAP
        summaries. ``"joint"`` samples one shared parameter set by summing block
        likelihoods.
    """

    reg = registry if registry is not None else build_default_registry()
    strategy = coerce_block_fit_strategy(
        block_fit_strategy,
        field_name="block_fit_strategy",
    )

    if strategy == "independent":
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
                likelihood_program=likelihood_program,
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
            fit_mode="independent",
            input_n_blocks=len(subject.blocks),
        )

    block_traces = tuple(get_block_trace(block) for block in subject.blocks)
    requirements = reg.get("model", model_component_id).requirements
    for trace in block_traces:
        assert_trace_compatible(trace, requirements)

    joint_likelihood = JointBlockLikelihoodProgram(
        block_traces=block_traces,
        likelihood_program=likelihood_program,
    )
    joint_posterior = sample_posterior_model_from_registry(
        subject.blocks[0],
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
        likelihood_program=joint_likelihood,
        random_seed=random_seed,
    )
    block_results = (
        MCMCBlockResult(
            block_id=JOINT_BLOCK_ID,
            n_trials=int(sum(block.n_trials for block in subject.blocks)),
            posterior_result=joint_posterior,
        ),
    )
    return MCMCSubjectResult(
        subject_id=subject.subject_id,
        block_results=block_results,
        total_map_log_likelihood=float(joint_posterior.map_candidate.log_likelihood),
        total_map_log_posterior=float(joint_posterior.map_candidate.log_posterior),
        mean_block_map_params=dict(joint_posterior.map_candidate.params),
        fit_mode="joint",
        input_n_blocks=len(subject.blocks),
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
    likelihood_program: LikelihoodProgram | None = None,
    random_seed: int | None = None,
    block_fit_strategy: BlockFitStrategy = "independent",
) -> MCMCStudyResult:
    """Sample posterior draws across all study subjects/blocks.

    Parameters
    ----------
    study : StudyData
        Study dataset.
    model_component_id : str
        Registered model component ID.
    prior_program : PriorProgram
        Prior evaluator.
    initial_params : Mapping[str, float]
        Initial constrained parameter values.
    n_samples : int
        Number of retained posterior draws.
    n_warmup : int, optional
        Warmup iterations.
    thin : int, optional
        Thinning interval.
    proposal_scales : Mapping[str, float] | None, optional
        Optional proposal scales.
    bounds : Mapping[str, tuple[float | None, float | None]] | None, optional
        Optional hard parameter bounds.
    model_kwargs : Mapping[str, Any] | None, optional
        Fixed model constructor kwargs.
    registry : PluginRegistry | None, optional
        Optional registry instance.
    likelihood_program : LikelihoodProgram | None, optional
        Optional likelihood evaluator.
    random_seed : int | None, optional
        Random seed.
    block_fit_strategy : {"independent", "joint"}, optional
        Block handling strategy passed to :func:`sample_posterior_subject_data`.
    """

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
            likelihood_program=likelihood_program,
            random_seed=None if random_seed is None else int(random_seed) + index * 1000,
            block_fit_strategy=block_fit_strategy,
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
    "BlockFitStrategy",
    "MCMCBlockResult",
    "MCMCStudyResult",
    "MCMCSubjectResult",
    "sample_posterior_block_data",
    "sample_posterior_study_data",
    "sample_posterior_subject_data",
]
