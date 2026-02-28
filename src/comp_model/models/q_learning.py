"""Q-learning agent implementation.

The agent is generic to any problem that emits scalar reward in the outcome
payload (attribute ``reward`` or key ``"reward"``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from comp_model.core.contracts import DecisionContext
from comp_model.core.requirements import ComponentRequirements
from comp_model.plugins import ComponentManifest


@dataclass(frozen=True, slots=True)
class QLearningConfig:
    """Configuration for :class:`QLearningAgent`.

    Parameters
    ----------
    alpha : float
        Learning rate in ``[0, 1]``.
    beta : float
        Inverse-temperature for softmax policy. ``beta=0`` implies uniform
        probabilities over available actions.
    initial_value : float
        Initial Q-value for unseen actions.

    Raises
    ------
    ValueError
        If ``alpha`` is outside ``[0, 1]`` or if ``beta`` is negative.
    """

    alpha: float = 0.2
    beta: float = 3.0
    initial_value: float = 0.0

    def __post_init__(self) -> None:
        if self.alpha < 0.0 or self.alpha > 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.beta < 0.0:
            raise ValueError("beta must be >= 0")


class QLearningAgent:
    """Tabular Q-learning agent.

    Parameters
    ----------
    config : QLearningConfig | None, optional
        Hyperparameter bundle. Defaults are used when ``None``.
    reward_getter : Callable[[Any], float] | None, optional
        Optional custom outcome-to-reward extractor.

    Notes
    -----
    This implementation updates only the chosen action value via:

    ``Q[a] <- Q[a] + alpha * (reward - Q[a])``
    """

    def __init__(
        self,
        config: QLearningConfig | None = None,
        reward_getter: Callable[[Any], float] | None = None,
    ) -> None:
        self.config = config if config is not None else QLearningConfig()
        self._reward_getter = reward_getter if reward_getter is not None else _default_reward_getter
        self._q_values: dict[Any, float] = {}

    def start_episode(self) -> None:
        """Reset episode state.

        Notes
        -----
        Episode reset clears learned values by default. For warm-start behavior,
        create a separate model wrapper in future iterations.
        """

        self._q_values = {}

    def action_distribution(
        self,
        observation: Any,
        *,
        context: DecisionContext[Any],
    ) -> dict[Any, float]:
        """Compute softmax probabilities over available actions.

        Parameters
        ----------
        observation : Any
            Unused by tabular Q-learning in this baseline implementation.
        context : DecisionContext[Any]
            Per-trial context containing available actions.

        Returns
        -------
        dict[Any, float]
            Action probabilities keyed by available actions.
        """

        actions = context.available_actions
        for action in actions:
            self._q_values.setdefault(action, self.config.initial_value)

        if self.config.beta == 0.0:
            probability = 1.0 / float(len(actions))
            return {action: probability for action in actions}

        logits = np.asarray([self.config.beta * self._q_values[action] for action in actions], dtype=float)

        # Subtract max(logits) for numerically stable exponentiation.
        logits -= float(np.max(logits))
        exp_logits = np.exp(logits)
        probs = exp_logits / float(np.sum(exp_logits))
        return {action: float(prob) for action, prob in zip(actions, probs, strict=True)}

    def update(
        self,
        observation: Any,
        action: Any,
        outcome: Any,
        *,
        context: DecisionContext[Any],
    ) -> None:
        """Update the chosen action value.

        Parameters
        ----------
        observation : Any
            Unused in this baseline implementation.
        action : Any
            Selected action.
        outcome : Any
            Outcome object containing reward signal.
        context : DecisionContext[Any]
            Per-trial context.
        """

        if action not in context.available_actions:
            raise ValueError(f"action {action!r} not in available actions")

        self._q_values.setdefault(action, self.config.initial_value)
        reward = float(self._reward_getter(outcome))
        current = self._q_values[action]
        self._q_values[action] = current + self.config.alpha * (reward - current)

    def q_values_snapshot(self) -> dict[Any, float]:
        """Return a copy of current learned action values.

        Returns
        -------
        dict[Any, float]
            Action-value mapping.
        """

        return dict(self._q_values)


def _default_reward_getter(outcome: Any) -> float:
    """Extract scalar reward from outcome payload.

    Parameters
    ----------
    outcome : Any
        Outcome object. Supported forms are objects with ``reward`` attribute or
        mappings containing a ``"reward"`` key.

    Returns
    -------
    float
        Scalar reward value.

    Raises
    ------
    TypeError
        If reward cannot be extracted.
    """

    if hasattr(outcome, "reward"):
        return float(getattr(outcome, "reward"))

    if isinstance(outcome, Mapping) and "reward" in outcome:
        return float(outcome["reward"])

    raise TypeError(
        "Outcome must expose a reward via attribute 'reward' or mapping key 'reward'"
    )


def create_q_learning_agent(
    *,
    alpha: float = 0.2,
    beta: float = 3.0,
    initial_value: float = 0.0,
) -> QLearningAgent:
    """Factory used by plugin discovery.

    Parameters
    ----------
    alpha : float, optional
        Learning rate.
    beta : float, optional
        Inverse-temperature.
    initial_value : float, optional
        Initial value for unseen actions.

    Returns
    -------
    QLearningAgent
        Configured model instance.
    """

    config = QLearningConfig(alpha=alpha, beta=beta, initial_value=initial_value)
    return QLearningAgent(config=config)


PLUGIN_MANIFESTS = [
    ComponentManifest(
        kind="model",
        component_id="q_learning",
        factory=create_q_learning_agent,
        description="Tabular Q-learning with softmax policy",
        requirements=ComponentRequirements(required_outcome_fields=("reward",)),
    )
]
