"""Top-level package for ``comp_model``.

The package is organized around a generic decision loop:

1. a :class:`~comp_model.core.contracts.DecisionProblem` generates observations,
2. an :class:`~comp_model.core.contracts.AgentModel` chooses actions,
3. the problem returns outcomes,
4. the agent updates internal memory.

For multi-phase tasks, use
:func:`~comp_model.runtime.engine.run_trial_program` with a
:class:`~comp_model.runtime.program.TrialProgram`.

Notes
-----
Bandit is intentionally implemented as a specific problem under
``comp_model.problems`` and is not used as the central abstraction.
"""

from .core.contracts import AgentModel, DecisionContext, DecisionProblem
from .core.data import StudyData
from .recovery import (
    load_json_config,
    run_model_recovery,
    run_model_recovery_from_config,
    run_parameter_recovery,
    run_parameter_recovery_from_config,
)
from .runtime.engine import SimulationConfig, run_episode, run_trial_program

__all__ = [
    "AgentModel",
    "DecisionContext",
    "DecisionProblem",
    "SimulationConfig",
    "StudyData",
    "load_json_config",
    "run_model_recovery",
    "run_model_recovery_from_config",
    "run_parameter_recovery",
    "run_parameter_recovery_from_config",
    "run_episode",
    "run_trial_program",
]
