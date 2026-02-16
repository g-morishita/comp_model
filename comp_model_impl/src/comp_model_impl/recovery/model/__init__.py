"""
Model recovery subpackage.

Model recovery evaluates whether candidate models can be distinguished and
recovered under a given study design by simulating data from each generating
model and fitting all candidates, then selecting a winner under a chosen model
selection criterion (e.g., log-likelihood, AIC, BIC, WAIC).

Main entry points
-----------------
- :func:`comp_model_impl.recovery.model.run.run_model_recovery`
- :func:`comp_model_impl.recovery.model.analysis.confusion_matrix`

See Also
--------
comp_model_impl.recovery.parameter
    Parameter recovery subpackage.
"""
from __future__ import annotations

from .config import (
    ModelRecoveryConfig,
    ModelRecoveryComponents,
    ComponentSpec,
    GeneratingModelSpec,
    CandidateModelSpec,
    SelectionSpec,
    OutputSpec,
    DistSpec,
    SamplingSpec,
    load_model_recovery_config,
    config_to_json,
)
from .criteria import (
    ModelCriterion,
    LogLikelihoodCriterion,
    AICCriterion,
    BICCriterion,
    WAICCriterion,
    get_criterion,
)
from .run import (
    ModelRecoveryOutputs,
    RuntimeGeneratingModelSpec,
    RuntimeCandidateModelSpec,
    run_model_recovery,
)
from .analysis import (
    confusion_matrix,
    recovery_rates,
    summarize_delta_scores,
)

__all__ = [
    "ModelRecoveryConfig",
    "ModelRecoveryComponents",
    "ComponentSpec",
    "GeneratingModelSpec",
    "CandidateModelSpec",
    "SelectionSpec",
    "OutputSpec",
    "DistSpec",
    "SamplingSpec",
    "load_model_recovery_config",
    "config_to_json",
    "ModelCriterion",
    "LogLikelihoodCriterion",
    "AICCriterion",
    "BICCriterion",
    "WAICCriterion",
    "get_criterion",
    "ModelRecoveryOutputs",
    "RuntimeGeneratingModelSpec",
    "RuntimeCandidateModelSpec",
    "run_model_recovery",
    "confusion_matrix",
    "recovery_rates",
    "summarize_delta_scores",
]
