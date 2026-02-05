"""Parameter recovery utilities.

This package provides end-to-end utilities for parameter recovery:

- configuration loading and validation
- sampling of ground-truth parameters
- simulation and fitting orchestration
- metrics helpers

Modules
-------
- :mod:`comp_model_impl.recovery.parameter.config` — configuration schemas and loaders
- :mod:`comp_model_impl.recovery.parameter.sampling` — sampling of true parameters
- :mod:`comp_model_impl.recovery.parameter.run` — orchestration and output writing
- :mod:`comp_model_impl.recovery.parameter.analysis` — recovery metrics

Examples
--------
Load a recovery config and run a single recovery pass:

>>> from comp_model_impl.recovery.parameter import load_parameter_recovery_config, run_parameter_recovery
>>> # cfg = load_parameter_recovery_config("recovery_config.yaml")  # doctest: +SKIP
>>> # outputs = run_parameter_recovery(config=cfg, generator=..., model=..., estimator=...)  # doctest: +SKIP

Compute summary metrics:

>>> from comp_model_impl.recovery.parameter import compute_parameter_recovery_metrics
>>> # metrics = compute_parameter_recovery_metrics(outputs)  # doctest: +SKIP

See Also
--------
comp_model_impl.recovery.parameter.run
comp_model_impl.recovery.parameter.analysis
"""

from .config import (
    DistSpec,
    SamplingSpec,
    OutputSpec,
    ParameterRecoveryConfig,
    load_parameter_recovery_config,
)
from .run import ParameterRecoveryOutputs, run_parameter_recovery
from .analysis import compute_parameter_recovery_metrics

__all__ = [
    "DistSpec",
    "SamplingSpec",
    "OutputSpec",
    "ParameterRecoveryConfig",
    "load_parameter_recovery_config",
    "ParameterRecoveryOutputs",
    "run_parameter_recovery",
    "compute_parameter_recovery_metrics",
]
