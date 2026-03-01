"""Study-level fitting workflows built on top of reusable model fitting."""

from __future__ import annotations

from dataclasses import dataclass

from comp_model.core.data import BlockData, StudyData, SubjectData
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
    """

    subject_id: str
    block_results: tuple[BlockFitResult, ...]
    total_log_likelihood: float
    mean_best_params: dict[str, float]


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
) -> SubjectFitResult:
    """Fit one model independently to all blocks of one subject."""

    reg = registry if registry is not None else build_default_registry()
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
    )


def fit_study_data(
    study: StudyData,
    *,
    model_component_id: str,
    fit_spec: FitSpec,
    model_kwargs: dict[str, object] | None = None,
    registry: PluginRegistry | None = None,
    likelihood_program: LikelihoodProgram | None = None,
) -> StudyFitResult:
    """Fit one model independently to all subjects and blocks in a study."""

    reg = registry if registry is not None else build_default_registry()
    subject_results = tuple(
        fit_subject_data(
            subject,
            model_component_id=model_component_id,
            fit_spec=fit_spec,
            model_kwargs=model_kwargs,
            registry=reg,
            likelihood_program=likelihood_program,
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
    "BlockFitResult",
    "StudyFitResult",
    "SubjectFitResult",
    "fit_block_data",
    "fit_study_data",
    "fit_subject_data",
]
