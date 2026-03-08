"""Maximum-likelihood fitting primitives, wrappers, and config entrypoints."""

from ..component_config import ModelComponentSpec, model_component_spec_from_config
from .config import (
    fit_block_from_config,
    fit_study_from_config,
    fit_subject_from_config,
    fit_trace_from_config,
    mle_fit_spec_from_config,
)
from .estimators import (
    GridSearchMLEEstimator,
    MLECandidate,
    MLEFitResult,
    ScipyMinimizeDiagnostics,
    ScipyMinimizeMLEEstimator,
    TransformedScipyMinimizeMLEEstimator,
)
from .fitting import (
    MLEFitSpec,
    MLESolverType,
    coerce_episode_trace,
    coerce_episode_traces,
    fit_joint_traces,
    fit_joint_traces_from_registry,
    fit_trace,
    fit_trace_from_registry,
)
from .group import (
    BlockFitResult,
    StudyFitResult,
    SubjectFitResult,
    fit_block,
    fit_study,
    fit_subject,
)

__all__ = [
    "BlockFitResult",
    "GridSearchMLEEstimator",
    "MLEFitSpec",
    "MLECandidate",
    "MLEFitResult",
    "MLESolverType",
    "ModelComponentSpec",
    "ScipyMinimizeDiagnostics",
    "ScipyMinimizeMLEEstimator",
    "StudyFitResult",
    "SubjectFitResult",
    "TransformedScipyMinimizeMLEEstimator",
    "coerce_episode_trace",
    "coerce_episode_traces",
    "fit_block",
    "fit_block_from_config",
    "fit_joint_traces",
    "fit_joint_traces_from_registry",
    "fit_study",
    "fit_study_from_config",
    "fit_subject",
    "fit_subject_from_config",
    "fit_trace",
    "fit_trace_from_config",
    "fit_trace_from_registry",
    "mle_fit_spec_from_config",
    "model_component_spec_from_config",
]
