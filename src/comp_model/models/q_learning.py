"""Asocial Q-value softmax model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from comp_model.core.contracts import DecisionContext
from comp_model.core.requirements import ComponentRequirements
from comp_model.plugins import ComponentManifest


@dataclass(frozen=True, slots=True)
class AsocialQValueSoftmaxConfig:
    """Configuration for :class:`AsocialQValueSoftmaxModel`.

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


class AsocialQValueSoftmaxModel:
    """Asocial chosen-only Q-learning model with softmax policy.

    Model Contract
    --------------
    Decision Rule
        For action values ``Q[a]`` and inverse temperature ``beta``:
        ``P(a) = softmax(beta * Q[a])`` over currently available actions.
    Update Rule
        After choosing action ``a`` and observing reward ``r``:
        ``Q[a] <- Q[a] + alpha * (r - Q[a])``.
        Only the chosen action value is updated.

    Parameters
    ----------
    config : AsocialQValueSoftmaxConfig | None, optional
        Hyperparameter bundle. Defaults are used when ``None``.
    reward_getter : Callable[[Any], float] | None, optional
        Optional custom outcome-to-reward extractor.
    """

    def __init__(
        self,
        config: AsocialQValueSoftmaxConfig | None = None,
        reward_getter: Callable[[Any], float] | None = None,
    ) -> None:
        self.config = config if config is not None else AsocialQValueSoftmaxConfig()
        self._reward_getter = reward_getter if reward_getter is not None else _default_reward_getter
        self._q_values: dict[Any, float] = {}

    def start_episode(self) -> None:
        """Reset episode state.

        Notes
        -----
        Episode reset clears learned values by default.
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
            Unused by this baseline implementation.
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


def create_asocial_q_value_softmax_model(
    *,
    alpha: float = 0.2,
    beta: float = 3.0,
    initial_value: float = 0.0,
) -> AsocialQValueSoftmaxModel:
    """Factory used by plugin discovery for canonical Q-learning model.

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
    AsocialQValueSoftmaxModel
        Configured model instance.
    """

    config = AsocialQValueSoftmaxConfig(alpha=alpha, beta=beta, initial_value=initial_value)
    return AsocialQValueSoftmaxModel(config=config)


PLUGIN_MANIFESTS = [
    ComponentManifest(
        kind="model",
        component_id="asocial_q_value_softmax",
        factory=create_asocial_q_value_softmax_model,
        description="Asocial tabular Q-learning with softmax policy",
        requirements=ComponentRequirements(required_outcome_fields=("reward",)),
    )
]
