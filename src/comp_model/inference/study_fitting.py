"""Study-level fitting workflows built on top of reusable model fitting."""

from __future__ import annotations

from dataclasses import dataclass

from comp_model.core.data import BlockData, StudyData, SubjectData, get_block_trace
from comp_model.inference.block_strategy import (
    JOINT_BLOCK_ID,
    BlockFitStrategy,
    JointBlockLikelihoodProgram,
    coerce_block_fit_strategy,
)
from comp_model.inference.compatibility import assert_trace_compatible
from comp_model.inference.fitting import FitSpec, fit_model_from_registry
from comp_model.inference.likelihood import LikelihoodProgram
from comp_model.inference.mle import MLEFitResult
from comp_model.plugins import PluginRegistry, build_default_registry


@dataclass(frozen=True, slots=True)
class BlockFitResult:
    """Fit output for one block.

    Parameters
    ----------
    block_id : str | int | None
        Block identifier when available.
    n_trials : int
        Number of trials in the block.
    fit_result : MLEFitResult
        Model fit result for this block.
    """

    block_id: str | int | None
    n_trials: int
    fit_result: MLEFitResult


@dataclass(frozen=True, slots=True)
class SubjectFitResult:
    """Fit output for one subject across all blocks.

    Parameters
    ----------
    subject_id : str
        Subject identifier.
    block_results : tuple[BlockFitResult, ...]
        Per-block fit results.
    total_log_likelihood : float
        Sum of best block-level log-likelihood values.
    mean_best_params : dict[str, float]
        Mean best-fit parameter values across blocks for keys shared by all
        block best-parameter mappings.
    fit_mode : {"independent", "joint"}
        Block-fitting strategy used for this subject.
    input_n_blocks : int | None
        Number of blocks in the original input subject data.
    """

    subject_id: str
    block_results: tuple[BlockFitResult, ...]
    total_log_likelihood: float
    mean_best_params: dict[str, float]
    fit_mode: BlockFitStrategy = "independent"
    input_n_blocks: int | None = None


@dataclass(frozen=True, slots=True)
class StudyFitResult:
    """Fit output for a full study.

    Parameters
    ----------
    subject_results : tuple[SubjectFitResult, ...]
        Per-subject fit summaries.
    total_log_likelihood : float
        Sum of subject-level total log-likelihood values.
    """

    subject_results: tuple[SubjectFitResult, ...]
    total_log_likelihood: float

    @property
    def n_subjects(self) -> int:
        """Return number of fitted subjects."""

        return len(self.subject_results)


def fit_block_data(
    block: BlockData,
    *,
    model_component_id: str,
    fit_spec: FitSpec,
    model_kwargs: dict[str, object] | None = None,
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> BlockFitResult:
    """Fit one model to one block.

    Parameters
    ----------
    block : BlockData
        Block dataset.
    model_component_id : str
        Registered model component ID.
    fit_spec : FitSpec
        Estimator specification.
    model_kwargs : dict[str, object] | None, optional
        Fixed model constructor kwargs.
    registry : PluginRegistry | None, optional
        Optional registry instance.

    Returns
    -------
    BlockFitResult
        Block-level fit summary.
    """

    fit = fit_model_from_registry(
        block,
        model_component_id=model_component_id,
        fit_spec=fit_spec,
        model_kwargs=model_kwargs,
        registry=registry,
        likelihood_program=likelihood_program,
    )
    return BlockFitResult(
        block_id=block.block_id,
        n_trials=block.n_trials,
        fit_result=fit,
    )


def fit_subject_data(
    subject: SubjectData,
    *,
    model_component_id: str,
    fit_spec: FitSpec,
    model_kwargs: dict[str, object] | None = None,
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
    block_fit_strategy: BlockFitStrategy = "independent",
) -> SubjectFitResult:
    """Fit one model across all blocks of one subject.

    Parameters
    ----------
    subject : SubjectData
        Subject dataset.
    model_component_id : str
        Registered model component ID.
    fit_spec : FitSpec
        Estimator specification.
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
            fit_block_data(
                block,
                model_component_id=model_component_id,
                fit_spec=fit_spec,
                model_kwargs=model_kwargs,
                registry=reg,
                likelihood_program=likelihood_program,
            )
            for block in subject.blocks
        )
        total_log_likelihood = float(
            sum(block.fit_result.best.log_likelihood for block in block_results)
        )
        mean_best_params = _mean_params_across_block_best(block_results)
        return SubjectFitResult(
            subject_id=subject.subject_id,
            block_results=block_results,
            total_log_likelihood=total_log_likelihood,
            mean_best_params=mean_best_params,
            fit_mode="independent",
            input_n_blocks=len(subject.blocks),
        )

    block_traces = tuple(get_block_trace(block) for block in subject.blocks)
    requirements = reg.get("model", model_component_id).requirements
    if requirements is not None:
        for trace in block_traces:
            assert_trace_compatible(trace, requirements)

    joint_likelihood = JointBlockLikelihoodProgram(
        block_traces=block_traces,
        likelihood_program=likelihood_program,
    )
    joint_fit = fit_model_from_registry(
        subject.blocks[0],
        model_component_id=model_component_id,
        fit_spec=fit_spec,
        model_kwargs=model_kwargs,
        registry=reg,
        likelihood_program=joint_likelihood,
    )
    block_results = (
        BlockFitResult(
            block_id=JOINT_BLOCK_ID,
            n_trials=int(sum(block.n_trials for block in subject.blocks)),
            fit_result=joint_fit,
        ),
    )

    return SubjectFitResult(
        subject_id=subject.subject_id,
        block_results=block_results,
        total_log_likelihood=float(joint_fit.best.log_likelihood),
        mean_best_params=dict(joint_fit.best.params),
        fit_mode="joint",
        input_n_blocks=len(subject.blocks),
    )


def fit_study_data(
    study: StudyData,
    *,
    model_component_id: str,
    fit_spec: FitSpec,
    model_kwargs: dict[str, object] | None = None,
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
    block_fit_strategy: BlockFitStrategy = "independent",
) -> StudyFitResult:
    """Fit one model to all subjects and blocks in a study.

    Parameters
    ----------
    study : StudyData
        Study dataset.
    model_component_id : str
        Registered model component ID.
    fit_spec : FitSpec
        Estimator specification.
    model_kwargs : dict[str, object] | None, optional
        Fixed model constructor kwargs.
    registry : PluginRegistry | None, optional
        Optional registry instance.
    likelihood_program : LikelihoodProgram | None, optional
        Optional likelihood evaluator.
    block_fit_strategy : {"independent", "joint"}, optional
        Block handling strategy passed to :func:`fit_subject_data`.
    """

    reg = registry if registry is not None else build_default_registry()
    subject_results = tuple(
        fit_subject_data(
            subject,
            model_component_id=model_component_id,
            fit_spec=fit_spec,
            model_kwargs=model_kwargs,
            registry=reg,
            likelihood_program=likelihood_program,
            block_fit_strategy=block_fit_strategy,
        )
        for subject in study.subjects
    )

    total_log_likelihood = float(
        sum(subject.total_log_likelihood for subject in subject_results)
    )
    return StudyFitResult(
        subject_results=subject_results,
        total_log_likelihood=total_log_likelihood,
    )


def _mean_params_across_block_best(block_results: tuple[BlockFitResult, ...]) -> dict[str, float]:
    """Average shared best-fit parameters across block fits."""

    if not block_results:
        return {}

    shared_keys: set[str] | None = None
    for block in block_results:
        keys = set(block.fit_result.best.params)
        shared_keys = keys if shared_keys is None else (shared_keys & keys)

    if not shared_keys:
        return {}

    out: dict[str, float] = {}
    for key in sorted(shared_keys):
        values = [float(block.fit_result.best.params[key]) for block in block_results]
        out[key] = float(sum(values) / len(values))
    return out


__all__ = [
    "BlockFitStrategy",
    "BlockFitResult",
    "StudyFitResult",
    "SubjectFitResult",
    "fit_block_data",
    "fit_study_data",
    "fit_subject_data",
]
