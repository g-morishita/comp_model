"""Uniform-random baseline model."""

from __future__ import annotations

from typing import Any

from comp_model.core.contracts import DecisionContext
from comp_model.plugins import ComponentManifest


class UniformRandomPolicyModel:
    """Uniform-random policy model.

    Model Contract
    --------------
    Decision Rule
        For ``N`` available actions, assign ``1 / N`` probability to each
        available action.
    Update Rule
        No-op. This model does not maintain latent state.

    Methods follow the :class:`comp_model.core.contracts.AgentModel` protocol.
    """

    def start_episode(self) -> None:
        """Reset episode state.

        Notes
        -----
        This model has no persistent episode state.
        """

    def action_distribution(
        self,
        observation: Any,
        *,
        context: DecisionContext[Any],
    ) -> dict[Any, float]:
        """Return a uniform distribution over available actions.

        Parameters
        ----------
        observation : Any
            Unused observation object.
        context : DecisionContext[Any]
            Per-trial context containing available actions.

        Returns
        -------
        dict[Any, float]
            Uniform action probabilities.
        """

        probability = 1.0 / float(len(context.available_actions))
        return {action: probability for action in context.available_actions}

    def update(
        self,
        observation: Any,
        action: Any,
        outcome: Any,
        *,
        context: DecisionContext[Any],
    ) -> None:
        """No-op update.

        Parameters
        ----------
        observation : Any
            Unused.
        action : Any
            Unused.
        outcome : Any
            Unused.
        context : DecisionContext[Any]
            Unused.
        """


def create_uniform_random_policy_model() -> UniformRandomPolicyModel:
    """Factory used by plugin discovery for the canonical random model.

    Returns
    -------
    UniformRandomPolicyModel
        Uniform-random baseline model.
    """

    return UniformRandomPolicyModel()


PLUGIN_MANIFESTS = [
    ComponentManifest(
        kind="model",
        component_id="uniform_random_policy",
        factory=create_uniform_random_policy_model,
        description="Uniform random policy baseline model",
    )
]
