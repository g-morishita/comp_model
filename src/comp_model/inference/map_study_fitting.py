"""Study-level MAP fitting workflows."""

from __future__ import annotations

from dataclasses import dataclass

from comp_model.core.data import BlockData, StudyData, SubjectData, get_block_trace
from comp_model.inference.bayes import (
    BayesFitResult,
    MapFitSpec,
    PriorProgram,
    fit_map_model_from_registry,
)
from comp_model.inference.block_strategy import (
    JOINT_BLOCK_ID,
    BlockFitStrategy,
    JointBlockLikelihoodProgram,
    coerce_block_fit_strategy,
)
from comp_model.inference.compatibility import assert_trace_compatible
from comp_model.inference.likelihood import LikelihoodProgram
from comp_model.plugins import PluginRegistry, build_default_registry


@dataclass(frozen=True, slots=True)
class MapBlockFitResult:
    """MAP fit output for one block.

    Parameters
    ----------
    block_id : str | int | None
        Block identifier.
    n_trials : int
        Number of trials in this block.
    fit_result : BayesFitResult
        MAP fit result for this block.
    """

    block_id: str | int | None
    n_trials: int
    fit_result: BayesFitResult


@dataclass(frozen=True, slots=True)
class MapSubjectFitResult:
    """MAP fit output for one subject across blocks.

    Parameters
    ----------
    subject_id : str
        Subject identifier.
    block_results : tuple[MapBlockFitResult, ...]
        Per-block MAP fit results.
    total_log_likelihood : float
        Sum of block-level MAP log-likelihood values.
    total_log_posterior : float
        Sum of block-level MAP log-posterior values.
    mean_map_params : dict[str, float]
        Mean MAP parameter values across shared parameter keys.
    fit_mode : {"independent", "joint"}
        Block-fitting strategy used for this subject.
    input_n_blocks : int | None
        Number of blocks in the original input subject data.
    """

    subject_id: str
    block_results: tuple[MapBlockFitResult, ...]
    total_log_likelihood: float
    total_log_posterior: float
    mean_map_params: dict[str, float]
    fit_mode: BlockFitStrategy = "independent"
    input_n_blocks: int | None = None


@dataclass(frozen=True, slots=True)
class MapStudyFitResult:
    """MAP fit output for a full study.

    Parameters
    ----------
    subject_results : tuple[MapSubjectFitResult, ...]
        Per-subject MAP fit summaries.
    total_log_likelihood : float
        Sum of subject-level total log-likelihood values.
    total_log_posterior : float
        Sum of subject-level total log-posterior values.
    """

    subject_results: tuple[MapSubjectFitResult, ...]
    total_log_likelihood: float
    total_log_posterior: float

    @property
    def n_subjects(self) -> int:
        """Return number of fitted subjects."""

        return len(self.subject_results)


def fit_map_block_data(
    block: BlockData,
    *,
    model_component_id: str,
    prior_program: PriorProgram,
    fit_spec: MapFitSpec,
    model_kwargs: dict[str, object] | None = None,
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> MapBlockFitResult:
    """Fit one model with MAP to one block."""

    fit = fit_map_model_from_registry(
        block,
        model_component_id=model_component_id,
        prior_program=prior_program,
        fit_spec=fit_spec,
        model_kwargs=model_kwargs,
        registry=registry,
        likelihood_program=likelihood_program,
    )
    return MapBlockFitResult(
        block_id=block.block_id,
        n_trials=block.n_trials,
        fit_result=fit,
    )


def fit_map_subject_data(
    subject: SubjectData,
    *,
    model_component_id: str,
    prior_program: PriorProgram,
    fit_spec: MapFitSpec,
    model_kwargs: dict[str, object] | None = None,
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
    block_fit_strategy: BlockFitStrategy = "independent",
) -> MapSubjectFitResult:
    """Fit one model with MAP across all blocks for one subject.

    Parameters
    ----------
    subject : SubjectData
        Subject dataset.
    model_component_id : str
        Registered model component ID.
    prior_program : PriorProgram
        Prior evaluator.
    fit_spec : MapFitSpec
        MAP estimator specification.
    model_kwargs : dict[str, object] | None, optional
        Fixed model constructor kwargs.
    registry : PluginRegistry | None, optional
        Optional registry instance.
    likelihood_program : LikelihoodProgram | None, optional
        Optional likelihood evaluator.
    block_fit_strategy : {"independent", "joint"}, optional
        ``"independent"`` fits each block separately and aggregates fit
        summaries. ``"joint"`` fits one shared parameter set by summing block
        likelihoods.
    """

    reg = registry if registry is not None else build_default_registry()
    strategy = coerce_block_fit_strategy(
        block_fit_strategy,
        field_name="block_fit_strategy",
    )

    if strategy == "independent":
        block_results = tuple(
            fit_map_block_data(
                block,
                model_component_id=model_component_id,
                prior_program=prior_program,
                fit_spec=fit_spec,
                model_kwargs=model_kwargs,
                registry=reg,
                likelihood_program=likelihood_program,
            )
            for block in subject.blocks
        )
        total_log_likelihood = float(
            sum(block.fit_result.map_candidate.log_likelihood for block in block_results)
        )
        total_log_posterior = float(
            sum(block.fit_result.map_candidate.log_posterior for block in block_results)
        )
        mean_map_params = _mean_map_params_across_blocks(block_results)
        return MapSubjectFitResult(
            subject_id=subject.subject_id,
            block_results=block_results,
            total_log_likelihood=total_log_likelihood,
            total_log_posterior=total_log_posterior,
            mean_map_params=mean_map_params,
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
    joint_fit = fit_map_model_from_registry(
        subject.blocks[0],
        model_component_id=model_component_id,
        prior_program=prior_program,
        fit_spec=fit_spec,
        model_kwargs=model_kwargs,
        registry=reg,
        likelihood_program=joint_likelihood,
    )
    block_results = (
        MapBlockFitResult(
            block_id=JOINT_BLOCK_ID,
            n_trials=int(sum(block.n_trials for block in subject.blocks)),
            fit_result=joint_fit,
        ),
    )

    return MapSubjectFitResult(
        subject_id=subject.subject_id,
        block_results=block_results,
        total_log_likelihood=float(joint_fit.map_candidate.log_likelihood),
        total_log_posterior=float(joint_fit.map_candidate.log_posterior),
        mean_map_params=dict(joint_fit.map_candidate.params),
        fit_mode="joint",
        input_n_blocks=len(subject.blocks),
    )


def fit_map_study_data(
    study: StudyData,
    *,
    model_component_id: str,
    prior_program: PriorProgram,
    fit_spec: MapFitSpec,
    model_kwargs: dict[str, object] | None = None,
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
    block_fit_strategy: BlockFitStrategy = "independent",
) -> MapStudyFitResult:
    """Fit one model with MAP across all study subjects/blocks.

    Parameters
    ----------
    study : StudyData
        Study dataset.
    model_component_id : str
        Registered model component ID.
    prior_program : PriorProgram
        Prior evaluator.
    fit_spec : MapFitSpec
        MAP estimator specification.
    model_kwargs : dict[str, object] | None, optional
        Fixed model constructor kwargs.
    registry : PluginRegistry | None, optional
        Optional registry instance.
    likelihood_program : LikelihoodProgram | None, optional
        Optional likelihood evaluator.
    block_fit_strategy : {"independent", "joint"}, optional
        Block handling strategy passed to :func:`fit_map_subject_data`.
    """

    reg = registry if registry is not None else build_default_registry()
    subject_results = tuple(
        fit_map_subject_data(
            subject,
            model_component_id=model_component_id,
            prior_program=prior_program,
            fit_spec=fit_spec,
            model_kwargs=model_kwargs,
            registry=reg,
            likelihood_program=likelihood_program,
            block_fit_strategy=block_fit_strategy,
        )
        for subject in study.subjects
    )
    return MapStudyFitResult(
        subject_results=subject_results,
        total_log_likelihood=float(
            sum(item.total_log_likelihood for item in subject_results)
        ),
        total_log_posterior=float(
            sum(item.total_log_posterior for item in subject_results)
        ),
    )


def _mean_map_params_across_blocks(block_results: tuple[MapBlockFitResult, ...]) -> dict[str, float]:
    """Average shared MAP parameters across blocks."""

    if not block_results:
        return {}

    shared_keys: set[str] | None = None
    for block in block_results:
        keys = set(block.fit_result.map_candidate.params)
        shared_keys = keys if shared_keys is None else (shared_keys & keys)

    if not shared_keys:
        return {}

    out: dict[str, float] = {}
    for key in sorted(shared_keys):
        values = [float(block.fit_result.map_candidate.params[key]) for block in block_results]
        out[key] = float(sum(values) / len(values))
    return out


__all__ = [
    "BlockFitStrategy",
    "MapBlockFitResult",
    "MapStudyFitResult",
    "MapSubjectFitResult",
    "fit_map_block_data",
    "fit_map_study_data",
    "fit_map_subject_data",
]
