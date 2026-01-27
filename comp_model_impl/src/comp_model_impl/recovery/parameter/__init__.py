"""Parameter recovery utilities.

Includes config loading, sampling, run orchestration, and plotting helpers.
"""

from .config import (
    DistSpec,
    SamplingSpec,
    OutputSpec,
    PlotSpec,
    ParameterRecoveryConfig,
    load_parameter_recovery_config,
)
from .run import ParameterRecoveryOutputs, run_parameter_recovery
from .analysis import compute_parameter_recovery_metrics
from .plots import plot_parameter_recovery_scatter

__all__ = [
    "DistSpec",
    "SamplingSpec",
    "OutputSpec",
    "PlotSpec",
    "ParameterRecoveryConfig",
    "load_parameter_recovery_config",
    "ParameterRecoveryOutputs",
    "run_parameter_recovery",
    "compute_parameter_recovery_metrics",
    "plot_parameter_recovery_scatter",
]
