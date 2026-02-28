"""Top-level package for ``comp_model_v2``.

The package is organized around a generic decision loop:

1. a :class:`~comp_model_v2.core.contracts.DecisionProblem` generates observations,
2. an :class:`~comp_model_v2.core.contracts.AgentModel` chooses actions,
3. the problem returns outcomes,
4. the agent updates internal memory.

Notes
-----
Bandit is intentionally implemented as a specific problem under
``comp_model_v2.problems`` and is not used as the central abstraction.
"""

from .core.contracts import AgentModel, DecisionContext, DecisionProblem
from .runtime.engine import SimulationConfig, run_episode

__all__ = [
    "AgentModel",
    "DecisionContext",
    "DecisionProblem",
    "SimulationConfig",
    "run_episode",
]
