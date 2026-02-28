"""Asocial state-indexed Q-value model family.

This module defines canonical asocial model names and keeps v1-style class/ID
aliases as deprecations during migration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
import warnings

import numpy as np

from comp_model.core.contracts import DecisionContext
from comp_model.core.requirements import ComponentRequirements
from comp_model.plugins import ComponentManifest


# TODO(v0.3.0): Remove deprecated alias names and IDs.
def _warn_deprecated_alias(old_name: str, new_name: str) -> None:
    """Emit a standardized deprecation warning for model aliases."""

    warnings.warn(
        (
            f"{old_name} is deprecated and will be removed in v0.3.0. "
            f"Use {new_name} instead."
        ),
        DeprecationWarning,
        stacklevel=3,
    )


@dataclass(slots=True)
class AsocialStateQValueSoftmaxModel:
    """Asocial state-indexed chosen-only Q-learning model.

    Model Contract
    --------------
    Decision Rule
        Given latent state ``s`` and action-values ``Q[s, a]``:
        ``P(a | s) = softmax(beta * Q[s, a])`` over available actions.
    Update Rule
        For chosen action ``a`` and observed reward ``r``:
        ``Q[s, a] <- Q[s, a] + alpha * (r - Q[s, a])``.
        No update occurs when outcome is missing.

    Parameters
    ----------
    alpha : float, optional
        Learning rate in ``[0, 1]``.
    beta : float, optional
        Softmax inverse-temperature. ``beta=0`` yields uniform probabilities.
    initial_value : float, optional
        Initial Q-value for unseen state-action pairs.
    """

    alpha: float = 0.2
    beta: float = 5.0
    initial_value: float = 0.0
    _q_values: dict[int, dict[Any, float]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_alpha(self.alpha, name="alpha")
        _validate_beta(self.beta)

    def start_episode(self) -> None:
        """Reset latent values at the start of an episode."""

        self._q_values = {}

    def action_distribution(
        self,
        observation: Any,
        *,
        context: DecisionContext[Any],
    ) -> dict[Any, float]:
        """Return softmax action probabilities for current state."""

        state = _extract_state(observation)
        q_state = _ensure_q_state(self._q_values, state=state, actions=context.available_actions, initial=self.initial_value)
        utilities = np.asarray([q_state[action] for action in context.available_actions], dtype=float)
        probs = _softmax(utilities, beta=float(self.beta))
        return {action: float(prob) for action, prob in zip(context.available_actions, probs, strict=True)}

    def update(
        self,
        observation: Any,
        action: Any,
        outcome: Any,
        *,
        context: DecisionContext[Any],
    ) -> None:
        """Update the chosen action value when outcome is observed."""

        if action not in context.available_actions:
            raise ValueError(f"action {action!r} not in available_actions")

        if outcome is None:
            return

        state = _extract_state(observation)
        q_state = _ensure_q_state(self._q_values, state=state, actions=context.available_actions, initial=self.initial_value)
        reward = _extract_reward(outcome)
        q_state[action] += float(self.alpha) * (reward - q_state[action])

    def q_values_snapshot(self, *, state: int = 0) -> dict[Any, float]:
        """Return a copy of Q-values for one state."""

        return dict(self._q_values.get(int(state), {}))


@dataclass(slots=True)
class AsocialStateQValueSoftmaxPerseverationModel:
    """Asocial state-indexed Q-learning with perseveration bias.

    Model Contract
    --------------
    Decision Rule
        Utility for action ``a`` in state ``s``:
        ``u[a] = beta * Q[s, a] + kappa * I[a == last_choice[s]]``.
        ``P(a | s) = softmax(u[a])``.
    Update Rule
        ``last_choice[s]`` is updated on every valid action.
        If outcome is observed: ``Q[s, a] <- Q[s, a] + alpha * (r - Q[s, a])``.

    Parameters
    ----------
    alpha : float, optional
        Learning rate in ``[0, 1]``.
    beta : float, optional
        Softmax inverse-temperature for Q-value term.
    kappa : float, optional
        Additive utility bonus for repeating previous action.
    initial_value : float, optional
        Initial Q-value for unseen state-action pairs.
    """

    alpha: float = 0.2
    beta: float = 5.0
    kappa: float = 1.0
    initial_value: float = 0.0
    _q_values: dict[int, dict[Any, float]] = field(default_factory=dict, init=False, repr=False)
    _last_choice: dict[int, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_alpha(self.alpha, name="alpha")
        _validate_beta(self.beta)

    def start_episode(self) -> None:
        """Reset latent values and stay trackers."""

        self._q_values = {}
        self._last_choice = {}

    def action_distribution(
        self,
        observation: Any,
        *,
        context: DecisionContext[Any],
    ) -> dict[Any, float]:
        """Return softmax probabilities with perseveration utility bonus."""

        state = _extract_state(observation)
        q_state = _ensure_q_state(self._q_values, state=state, actions=context.available_actions, initial=self.initial_value)

        last_choice = self._last_choice.get(state)
        utilities: list[float] = []
        for action in context.available_actions:
            stay_bonus = float(self.kappa) if action == last_choice else 0.0
            utilities.append(float(self.beta) * q_state[action] + stay_bonus)

        probs = _softmax(np.asarray(utilities, dtype=float), beta=1.0)
        return {action: float(prob) for action, prob in zip(context.available_actions, probs, strict=True)}

    def update(
        self,
        observation: Any,
        action: Any,
        outcome: Any,
        *,
        context: DecisionContext[Any],
    ) -> None:
        """Track selected action and optionally update chosen Q-value."""

        if action not in context.available_actions:
            raise ValueError(f"action {action!r} not in available_actions")

        state = _extract_state(observation)
        self._last_choice[state] = action

        if outcome is None:
            return

        q_state = _ensure_q_state(self._q_values, state=state, actions=context.available_actions, initial=self.initial_value)
        reward = _extract_reward(outcome)
        q_state[action] += float(self.alpha) * (reward - q_state[action])

    def q_values_snapshot(self, *, state: int = 0) -> dict[Any, float]:
        """Return a copy of Q-values for one state."""

        return dict(self._q_values.get(int(state), {}))


@dataclass(slots=True)
class AsocialStateQValueSoftmaxSplitAlphaModel:
    """Asocial state-indexed Q-learning with split, non-identifiable alphas.

    Model Contract
    --------------
    Decision Rule
        ``P(a | s) = softmax(beta * Q[s, a])`` over available actions.
    Update Rule
        Effective learning rate is ``alpha_1 + alpha_2``:
        ``Q[s, a] <- Q[s, a] + (alpha_1 + alpha_2) * (r - Q[s, a])``.
        This parameterization is intentionally non-identifiable.

    Parameters
    ----------
    alpha_1 : float, optional
        First learning-rate component in ``[0, 1]``.
    alpha_2 : float, optional
        Second learning-rate component in ``[0, 1]``.
    beta : float, optional
        Softmax inverse-temperature.
    initial_value : float, optional
        Initial Q-value for unseen state-action pairs.
    """

    alpha_1: float = 0.2
    alpha_2: float = 0.2
    beta: float = 5.0
    initial_value: float = 0.0
    _q_values: dict[int, dict[Any, float]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_alpha(self.alpha_1, name="alpha_1")
        _validate_alpha(self.alpha_2, name="alpha_2")
        _validate_beta(self.beta)

    def start_episode(self) -> None:
        """Reset latent values at the start of an episode."""

        self._q_values = {}

    def action_distribution(
        self,
        observation: Any,
        *,
        context: DecisionContext[Any],
    ) -> dict[Any, float]:
        """Return softmax action probabilities for current state."""

        state = _extract_state(observation)
        q_state = _ensure_q_state(self._q_values, state=state, actions=context.available_actions, initial=self.initial_value)
        utilities = np.asarray([q_state[action] for action in context.available_actions], dtype=float)
        probs = _softmax(utilities, beta=float(self.beta))
        return {action: float(prob) for action, prob in zip(context.available_actions, probs, strict=True)}

    def update(
        self,
        observation: Any,
        action: Any,
        outcome: Any,
        *,
        context: DecisionContext[Any],
    ) -> None:
        """Update chosen value using effective learning rate ``alpha_1 + alpha_2``."""

        if action not in context.available_actions:
            raise ValueError(f"action {action!r} not in available_actions")

        if outcome is None:
            return

        state = _extract_state(observation)
        q_state = _ensure_q_state(self._q_values, state=state, actions=context.available_actions, initial=self.initial_value)
        reward = _extract_reward(outcome)

        effective_alpha = float(self.alpha_1) + float(self.alpha_2)
        q_state[action] += effective_alpha * (reward - q_state[action])

    def q_values_snapshot(self, *, state: int = 0) -> dict[Any, float]:
        """Return a copy of Q-values for one state."""

        return dict(self._q_values.get(int(state), {}))


class QRL(AsocialStateQValueSoftmaxModel):
    """Deprecated alias for :class:`AsocialStateQValueSoftmaxModel`."""

    def __init__(self, alpha: float = 0.2, beta: float = 5.0, initial_value: float = 0.0) -> None:
        _warn_deprecated_alias("QRL", "AsocialStateQValueSoftmaxModel")
        super().__init__(alpha=alpha, beta=beta, initial_value=initial_value)


class QRL_Stay(AsocialStateQValueSoftmaxPerseverationModel):
    """Deprecated alias for :class:`AsocialStateQValueSoftmaxPerseverationModel`."""

    def __init__(
        self,
        alpha: float = 0.2,
        beta: float = 5.0,
        kappa: float = 1.0,
        initial_value: float = 0.0,
    ) -> None:
        _warn_deprecated_alias("QRL_Stay", "AsocialStateQValueSoftmaxPerseverationModel")
        super().__init__(alpha=alpha, beta=beta, kappa=kappa, initial_value=initial_value)


class UnidentifiableQRL(AsocialStateQValueSoftmaxSplitAlphaModel):
    """Deprecated alias for :class:`AsocialStateQValueSoftmaxSplitAlphaModel`."""

    def __init__(
        self,
        alpha_1: float = 0.2,
        alpha_2: float = 0.2,
        beta: float = 5.0,
        initial_value: float = 0.0,
    ) -> None:
        _warn_deprecated_alias("UnidentifiableQRL", "AsocialStateQValueSoftmaxSplitAlphaModel")
        super().__init__(alpha_1=alpha_1, alpha_2=alpha_2, beta=beta, initial_value=initial_value)


def _extract_state(observation: Any) -> int:
    """Extract latent state index from observation payload.

    Parameters
    ----------
    observation : Any
        Observation object supplied to model policy/update.

    Returns
    -------
    int
        State index. Defaults to ``0`` when unavailable.
    """

    if isinstance(observation, Mapping) and "state" in observation:
        return int(observation["state"])
    return 0


def _extract_reward(outcome: Any) -> float:
    """Extract scalar reward from supported outcome payload forms."""

    if hasattr(outcome, "reward"):
        return float(getattr(outcome, "reward"))

    if isinstance(outcome, Mapping) and "reward" in outcome:
        return float(outcome["reward"])

    raise TypeError(
        "Outcome must expose reward via attribute 'reward' or mapping key 'reward'"
    )


def _ensure_q_state(
    q_values: dict[int, dict[Any, float]],
    *,
    state: int,
    actions: tuple[Any, ...],
    initial: float,
) -> dict[Any, float]:
    """Ensure a state's action values exist for all available actions."""

    state_values = q_values.setdefault(int(state), {})
    for action in actions:
        state_values.setdefault(action, float(initial))
    return state_values


def _softmax(values: np.ndarray, *, beta: float) -> np.ndarray:
    """Compute numerically stable softmax probabilities.

    Parameters
    ----------
    values : numpy.ndarray
        Utility vector.
    beta : float
        Inverse temperature scale.

    Returns
    -------
    numpy.ndarray
        Probability vector summing to one.
    """

    if values.size == 1:
        return np.asarray([1.0], dtype=float)

    if beta == 0.0:
        return np.ones(values.size, dtype=float) / float(values.size)

    logits = beta * np.asarray(values, dtype=float)
    logits -= float(np.max(logits))
    exp_logits = np.exp(logits)
    return exp_logits / float(np.sum(exp_logits))


def _validate_alpha(alpha: float, *, name: str) -> None:
    """Validate learning-rate parameter."""

    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def _validate_beta(beta: float) -> None:
    """Validate inverse-temperature parameter."""

    if beta < 0.0:
        raise ValueError("beta must be >= 0")


def create_asocial_state_q_value_softmax_model(
    *,
    alpha: float = 0.2,
    beta: float = 5.0,
    initial_value: float = 0.0,
) -> AsocialStateQValueSoftmaxModel:
    """Factory for :class:`AsocialStateQValueSoftmaxModel`."""

    return AsocialStateQValueSoftmaxModel(alpha=alpha, beta=beta, initial_value=initial_value)


def create_asocial_state_q_value_softmax_perseveration_model(
    *,
    alpha: float = 0.2,
    beta: float = 5.0,
    kappa: float = 1.0,
    initial_value: float = 0.0,
) -> AsocialStateQValueSoftmaxPerseverationModel:
    """Factory for :class:`AsocialStateQValueSoftmaxPerseverationModel`."""

    return AsocialStateQValueSoftmaxPerseverationModel(
        alpha=alpha,
        beta=beta,
        kappa=kappa,
        initial_value=initial_value,
    )


def create_asocial_state_q_value_softmax_split_alpha_model(
    *,
    alpha_1: float = 0.2,
    alpha_2: float = 0.2,
    beta: float = 5.0,
    initial_value: float = 0.0,
) -> AsocialStateQValueSoftmaxSplitAlphaModel:
    """Factory for :class:`AsocialStateQValueSoftmaxSplitAlphaModel`."""

    return AsocialStateQValueSoftmaxSplitAlphaModel(
        alpha_1=alpha_1,
        alpha_2=alpha_2,
        beta=beta,
        initial_value=initial_value,
    )


def create_qrl(
    *,
    alpha: float = 0.2,
    beta: float = 5.0,
    initial_value: float = 0.0,
) -> AsocialStateQValueSoftmaxModel:
    """Deprecated factory alias for :func:`create_asocial_state_q_value_softmax_model`."""

    _warn_deprecated_alias("create_qrl", "create_asocial_state_q_value_softmax_model")
    return create_asocial_state_q_value_softmax_model(alpha=alpha, beta=beta, initial_value=initial_value)


def create_qrl_stay(
    *,
    alpha: float = 0.2,
    beta: float = 5.0,
    kappa: float = 1.0,
    initial_value: float = 0.0,
) -> AsocialStateQValueSoftmaxPerseverationModel:
    """Deprecated alias for :func:`create_asocial_state_q_value_softmax_perseveration_model`."""

    _warn_deprecated_alias(
        "create_qrl_stay",
        "create_asocial_state_q_value_softmax_perseveration_model",
    )
    return create_asocial_state_q_value_softmax_perseveration_model(
        alpha=alpha,
        beta=beta,
        kappa=kappa,
        initial_value=initial_value,
    )


def create_unidentifiable_qrl(
    *,
    alpha_1: float = 0.2,
    alpha_2: float = 0.2,
    beta: float = 5.0,
    initial_value: float = 0.0,
) -> AsocialStateQValueSoftmaxSplitAlphaModel:
    """Deprecated alias for :func:`create_asocial_state_q_value_softmax_split_alpha_model`."""

    _warn_deprecated_alias(
        "create_unidentifiable_qrl",
        "create_asocial_state_q_value_softmax_split_alpha_model",
    )
    return create_asocial_state_q_value_softmax_split_alpha_model(
        alpha_1=alpha_1,
        alpha_2=alpha_2,
        beta=beta,
        initial_value=initial_value,
    )


PLUGIN_MANIFESTS = [
    ComponentManifest(
        kind="model",
        component_id="asocial_state_q_value_softmax",
        factory=create_asocial_state_q_value_softmax_model,
        description="Asocial state-indexed chosen-only Q-learning with softmax policy",
        requirements=ComponentRequirements(required_outcome_fields=("reward",)),
    ),
    ComponentManifest(
        kind="model",
        component_id="asocial_state_q_value_softmax_perseveration",
        factory=create_asocial_state_q_value_softmax_perseveration_model,
        description="Asocial state-indexed Q-learning with perseveration bias",
        requirements=ComponentRequirements(required_outcome_fields=("reward",)),
    ),
    ComponentManifest(
        kind="model",
        component_id="asocial_state_q_value_softmax_split_alpha",
        factory=create_asocial_state_q_value_softmax_split_alpha_model,
        description="Asocial state-indexed split-alpha non-identifiable Q-learning",
        requirements=ComponentRequirements(required_outcome_fields=("reward",)),
    ),
    ComponentManifest(
        kind="model",
        component_id="qrl",
        factory=create_qrl,
        description="DEPRECATED alias of asocial_state_q_value_softmax",
        requirements=ComponentRequirements(required_outcome_fields=("reward",)),
    ),
    ComponentManifest(
        kind="model",
        component_id="qrl_stay",
        factory=create_qrl_stay,
        description="DEPRECATED alias of asocial_state_q_value_softmax_perseveration",
        requirements=ComponentRequirements(required_outcome_fields=("reward",)),
    ),
    ComponentManifest(
        kind="model",
        component_id="unidentifiable_qrl",
        factory=create_unidentifiable_qrl,
        description="DEPRECATED alias of asocial_state_q_value_softmax_split_alpha",
        requirements=ComponentRequirements(required_outcome_fields=("reward",)),
    ),
]
