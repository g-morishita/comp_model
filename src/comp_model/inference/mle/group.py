"""Study-level fitting workflows built on top of reusable model fitting."""

from __future__ import annotations

from dataclasses import dataclass

from comp_model.core.data import BlockData, StudyData, SubjectData, get_block_trace
from comp_model.plugins import PluginRegistry, build_default_registry

from ..block_strategy import (
    JOINT_BLOCK_ID,
    BlockFitStrategy,
    coerce_block_fit_strategy,
)
from ..likelihood import LikelihoodProgram
from .fitting import (
    MLEFitSpec,
    fit_joint_traces_from_registry,
    fit_trace_from_registry,
)
from .estimators import MLEFitResult


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
    shared_best_params : dict[str, float] | None
        Shared best-fit parameter values when one parameter set is fit across
        all blocks. ``None`` for independent block fits.
    fit_mode : {"independent", "joint"}
        Block-fitting strategy used for this subject.
    input_n_blocks : int | None
        Number of blocks in the original input subject data.
    """

    subject_id: str
    block_results: tuple[BlockFitResult, ...]
    total_log_likelihood: float
    shared_best_params: dict[str, float] | None = None
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


def fit_block(
    block: BlockData,
    *,
    model_component_id: str,
    fit_spec: MLEFitSpec,
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
    fit_spec : MLEFitSpec
        MLE estimator specification.
    model_kwargs : dict[str, object] | None, optional
        Fixed model constructor kwargs.
    registry : PluginRegistry | None, optional
        Optional registry instance.

    Returns
    -------
    BlockFitResult
        Block-level fit summary.
    """

    fit = fit_trace_from_registry(
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


def fit_subject(
    subject: SubjectData,
    *,
    model_component_id: str,
    fit_spec: MLEFitSpec,
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
    fit_spec : MLEFitSpec
        MLE estimator specification.
    model_kwargs : dict[str, object] | None, optional
        Fixed model constructor kwargs.
    registry : PluginRegistry | None, optional
        Optional registry instance.
    likelihood_program : LikelihoodProgram | None, optional
        Optional likelihood evaluator.
    block_fit_strategy : {"independent", "joint"}, optional
        ``"independent"`` fits each block separately and returns one block
        estimate per block. ``"joint"`` fits one shared parameter set by
        summing block likelihoods.
    """

    reg = registry if registry is not None else build_default_registry()
    strategy = coerce_block_fit_strategy(
        block_fit_strategy,
        field_name="block_fit_strategy",
    )

    if strategy == "independent":
        block_results = tuple(
            fit_block(
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
        return SubjectFitResult(
            subject_id=subject.subject_id,
            block_results=block_results,
            total_log_likelihood=total_log_likelihood,
            fit_mode="independent",
            input_n_blocks=len(subject.blocks),
        )

    block_traces = tuple(get_block_trace(block) for block in subject.blocks)
    joint_fit = fit_joint_traces_from_registry(
        block_traces,
        model_component_id=model_component_id,
        fit_spec=fit_spec,
        model_kwargs=model_kwargs,
        registry=reg,
        likelihood_program=likelihood_program,
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
        shared_best_params=dict(joint_fit.best.params),
        fit_mode="joint",
        input_n_blocks=len(subject.blocks),
    )


def fit_study(
    study: StudyData,
    *,
    model_component_id: str,
    fit_spec: MLEFitSpec,
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
    fit_spec : MLEFitSpec
        MLE estimator specification.
    model_kwargs : dict[str, object] | None, optional
        Fixed model constructor kwargs.
    registry : PluginRegistry | None, optional
        Optional registry instance.
    likelihood_program : LikelihoodProgram | None, optional
        Optional likelihood evaluator.
    block_fit_strategy : {"independent", "joint"}, optional
        Block handling strategy passed to :func:`fit_subject`.
    """

    reg = registry if registry is not None else build_default_registry()
    subject_results = tuple(
        fit_subject(
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

__all__ = [
    "BlockFitStrategy",
    "BlockFitResult",
    "StudyFitResult",
    "SubjectFitResult",
    "fit_block",
    "fit_study",
    "fit_subject",
]
