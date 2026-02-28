"""Inference interfaces for replay, compatibility checks, and MLE fitting."""

from .compatibility import CompatibilityReport, assert_trace_compatible, check_trace_compatibility
from .fitting import (
    EstimatorType,
    FitSpec,
    build_model_fit_function,
    coerce_episode_trace,
    fit_model,
    fit_model_from_registry,
)
from .likelihood import ActionReplayLikelihood, LikelihoodProgram
from .mle import (
    GridSearchMLEEstimator,
    MLECandidate,
    MLEFitResult,
    ScipyMinimizeDiagnostics,
    ScipyMinimizeMLEEstimator,
    TransformedScipyMinimizeMLEEstimator,
)
from .study_fitting import (
    BlockFitResult,
    StudyFitResult,
    SubjectFitResult,
    fit_block_data,
    fit_study_data,
    fit_subject_data,
)
from .transforms import ParameterTransform, identity_transform, positive_log_transform, unit_interval_logit_transform

__all__ = [
    "ActionReplayLikelihood",
    "BlockFitResult",
    "CompatibilityReport",
    "EstimatorType",
    "FitSpec",
    "GridSearchMLEEstimator",
    "LikelihoodProgram",
    "MLECandidate",
    "MLEFitResult",
    "ParameterTransform",
    "ScipyMinimizeDiagnostics",
    "ScipyMinimizeMLEEstimator",
    "StudyFitResult",
    "SubjectFitResult",
    "TransformedScipyMinimizeMLEEstimator",
    "assert_trace_compatible",
    "build_model_fit_function",
    "check_trace_compatibility",
    "coerce_episode_trace",
    "fit_block_data",
    "fit_model",
    "fit_model_from_registry",
    "fit_study_data",
    "fit_subject_data",
    "identity_transform",
    "positive_log_transform",
    "unit_interval_logit_transform",
]
