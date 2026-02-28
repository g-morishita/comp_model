"""Noisy best-arm demonstrator implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from comp_model.core.contracts import DecisionContext
from comp_model.plugins import ComponentManifest


class NoisyBestArmDemonstrator:
    """Demonstrator that favors highest expected-reward available action.

    Parameters
    ----------
    reward_probabilities : Sequence[float]
        Expected reward probability per action index.
    epsilon : float, optional
        Exploration weight in ``[0, 1]``. Higher values flatten the policy.

    Raises
    ------
    ValueError
        If probabilities are empty/out-of-bounds or epsilon is invalid.
    """

    def __init__(self, reward_probabilities: Sequence[float], *, epsilon: float = 0.1) -> None:
        if len(reward_probabilities) == 0:
            raise ValueError("reward_probabilities must include at least one action")
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError("epsilon must be in [0, 1]")

        values = tuple(float(value) for value in reward_probabilities)
        for value in values:
            if value < 0.0 or value > 1.0:
                raise ValueError("reward probabilities must be within [0, 1]")

        self._reward_probabilities = values
        self._epsilon = float(epsilon)

    def start_episode(self) -> None:
        """Reset episode state (no-op)."""

    def action_distribution(
        self,
        observation: Any,
        *,
        context: DecisionContext[Any],
    ) -> dict[Any, float]:
        """Return epsilon-greedy probabilities over available actions.

        Parameters
        ----------
        observation : Any
            Unused observation payload.
        context : DecisionContext[Any]
            Runtime decision context.

        Returns
        -------
        dict[Any, float]
            Epsilon-greedy action probabilities.

        Raises
        ------
        ValueError
            If available action IDs are invalid for configured probabilities.
        """

        del observation

        actions = context.available_actions
        try:
            action_values = {action: self._reward_probabilities[int(action)] for action in actions}
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("available actions must map to valid integer action indices") from exc

        n_actions = len(actions)
        if n_actions == 1:
            return {actions[0]: 1.0}

        best_value = max(action_values.values())
        best_actions = sorted(action for action, value in action_values.items() if value == best_value)
        best_action = best_actions[0]

        base = self._epsilon / float(n_actions)
        distribution = {action: base for action in actions}
        distribution[best_action] += 1.0 - self._epsilon
        return distribution

    def update(
        self,
        observation: Any,
        action: Any,
        outcome: Any,
        *,
        context: DecisionContext[Any],
    ) -> None:
        """No-op update hook for runtime compatibility."""

        del observation, action, outcome, context


def create_noisy_best_arm_demonstrator(
    *,
    reward_probabilities: Sequence[float],
    epsilon: float = 0.1,
) -> NoisyBestArmDemonstrator:
    """Factory used by plugin discovery."""

    return NoisyBestArmDemonstrator(reward_probabilities=reward_probabilities, epsilon=epsilon)


PLUGIN_MANIFESTS = [
    ComponentManifest(
        kind="demonstrator",
        component_id="noisy_best_arm_demonstrator",
        factory=create_noisy_best_arm_demonstrator,
        description="Epsilon-greedy demonstrator over expected best arm",
    )
]
