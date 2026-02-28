"""Social observed-outcome and value-shaping model family."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from comp_model.core.contracts import DecisionContext
from comp_model.core.requirements import ComponentRequirements
from comp_model.plugins import ComponentManifest

from .social_utils import (
    ensure_q_state,
    extract_reward,
    extract_state,
    is_demonstrator_stage,
    is_subject_stage,
    stable_softmax,
    validate_alpha,
    validate_beta,
)


@dataclass(slots=True)
class SocialObservedOutcomeQModel:
    """Social observed-outcome Q-learning model.

    Model Contract
    --------------
    Decision Rule
        ``P(a | s) = softmax(beta * Q[s, a])`` over available actions.
    Update Rule
        Only demonstrator-generated outcomes are used:
        ``Q[s, demo_a] <- Q[s, demo_a] + alpha_observed * (r_demo - Q[s, demo_a])``.
        Subject-generated outcomes do not update values.

    Parameters
    ----------
    alpha_observed : float, optional
        Learning rate for observed demonstrator outcomes in ``[0, 1]``.
    beta : float, optional
        Softmax inverse-temperature.
    initial_value : float, optional
        Initial Q-value for unseen state-action pairs.
    """

    alpha_observed: float = 0.2
    beta: float = 3.0
    initial_value: float = 0.0
    _q_values: dict[int, dict[Any, float]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        validate_alpha(self.alpha_observed, name="alpha_observed")
        validate_beta(self.beta)

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

        state = extract_state(observation)
        q_state = ensure_q_state(
            self._q_values,
            state=state,
            actions=context.available_actions,
            initial=self.initial_value,
        )
        utilities = np.asarray([float(self.beta) * q_state[action] for action in context.available_actions], dtype=float)
        probs = stable_softmax(utilities)
        return {action: float(prob) for action, prob in zip(context.available_actions, probs, strict=True)}

    def update(
        self,
        observation: Any,
        action: Any,
        outcome: Any,
        *,
        context: DecisionContext[Any],
    ) -> None:
        """Update Q-values from demonstrator outcomes only."""

        if action not in context.available_actions:
            raise ValueError(f"action {action!r} not in available_actions")

        if not is_demonstrator_stage(observation, outcome):
            return
        if outcome is None:
            return

        state = extract_state(observation)
        q_state = ensure_q_state(
            self._q_values,
            state=state,
            actions=context.available_actions,
            initial=self.initial_value,
        )
        reward = extract_reward(outcome)
        q_state[action] += float(self.alpha_observed) * (reward - q_state[action])

    def q_values_snapshot(self, *, state: int = 0) -> dict[Any, float]:
        """Return a copy of Q-values for one state."""

        return dict(self._q_values.get(int(state), {}))


@dataclass(slots=True)
class SocialObservedOutcomeQPerseverationModel:
    """Social observed-outcome Q-learning with self perseveration.

    Model Contract
    --------------
    Decision Rule
        ``u[a] = beta * Q[s, a] + kappa * I[a == last_self_choice[s]]`` and
        ``P(a | s) = softmax(u[a])``.
    Update Rule
        Demonstrator-stage outcomes update ``Q`` with ``alpha_observed``.
        Subject-stage actions update ``last_self_choice`` only.

    Parameters
    ----------
    alpha_observed : float, optional
        Learning rate for observed demonstrator outcomes in ``[0, 1]``.
    beta : float, optional
        Softmax inverse-temperature.
    kappa : float, optional
        Additive perseveration utility bonus.
    initial_value : float, optional
        Initial Q-value for unseen state-action pairs.
    """

    alpha_observed: float = 0.2
    beta: float = 3.0
    kappa: float = 0.0
    initial_value: float = 0.0
    _q_values: dict[int, dict[Any, float]] = field(default_factory=dict, init=False, repr=False)
    _last_self_choice: dict[int, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        validate_alpha(self.alpha_observed, name="alpha_observed")
        validate_beta(self.beta)

    def start_episode(self) -> None:
        """Reset latent values and perseveration trackers."""

        self._q_values = {}
        self._last_self_choice = {}

    def action_distribution(
        self,
        observation: Any,
        *,
        context: DecisionContext[Any],
    ) -> dict[Any, float]:
        """Return softmax action probabilities with perseveration term."""

        state = extract_state(observation)
        q_state = ensure_q_state(
            self._q_values,
            state=state,
            actions=context.available_actions,
            initial=self.initial_value,
        )
        last_choice = self._last_self_choice.get(state)
        utilities: list[float] = []
        for available_action in context.available_actions:
            stay_bonus = float(self.kappa) if available_action == last_choice else 0.0
            utilities.append(float(self.beta) * q_state[available_action] + stay_bonus)

        probs = stable_softmax(np.asarray(utilities, dtype=float))
        return {action: float(prob) for action, prob in zip(context.available_actions, probs, strict=True)}

    def update(
        self,
        observation: Any,
        action: Any,
        outcome: Any,
        *,
        context: DecisionContext[Any],
    ) -> None:
        """Update perseveration state and demonstrator-outcome Q-values."""

        if action not in context.available_actions:
            raise ValueError(f"action {action!r} not in available_actions")

        state = extract_state(observation)

        if is_subject_stage(observation, outcome):
            self._last_self_choice[state] = action
            return

        if not is_demonstrator_stage(observation, outcome):
            return
        if outcome is None:
            return

        q_state = ensure_q_state(
            self._q_values,
            state=state,
            actions=context.available_actions,
            initial=self.initial_value,
        )
        reward = extract_reward(outcome)
        q_state[action] += float(self.alpha_observed) * (reward - q_state[action])

    def q_values_snapshot(self, *, state: int = 0) -> dict[Any, float]:
        """Return a copy of Q-values for one state."""

        return dict(self._q_values.get(int(state), {}))


@dataclass(slots=True)
class SocialObservedOutcomeValueShapingModel:
    """Social observed-outcome model with value shaping.

    Model Contract
    --------------
    Decision Rule
        ``P(a | s) = softmax(beta * Q[s, a])``.
    Update Rule
        Demonstrator-stage updates are applied sequentially to demonstrator
        action ``demo_a``:
        ``Q <- Q + alpha_social * (pseudo_reward - Q)``
        ``Q <- Q + alpha_observed * (r_demo - Q)`` when outcome is observed.

        Subject-generated outcomes do not update values.

    Parameters
    ----------
    alpha_observed : float, optional
        Learning rate for observed demonstrator outcomes in ``[0, 1]``.
    alpha_social : float, optional
        Learning rate for social value shaping in ``[0, 1]``.
    beta : float, optional
        Softmax inverse-temperature.
    pseudo_reward : float, optional
        Pseudo-reward target for value shaping.
    initial_value : float, optional
        Initial Q-value for unseen state-action pairs.
    """

    alpha_observed: float = 0.2
    alpha_social: float = 0.2
    beta: float = 3.0
    pseudo_reward: float = 1.0
    initial_value: float = 0.0
    _q_values: dict[int, dict[Any, float]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        validate_alpha(self.alpha_observed, name="alpha_observed")
        validate_alpha(self.alpha_social, name="alpha_social")
        validate_beta(self.beta)

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

        state = extract_state(observation)
        q_state = ensure_q_state(
            self._q_values,
            state=state,
            actions=context.available_actions,
            initial=self.initial_value,
        )
        utilities = np.asarray([float(self.beta) * q_state[action] for action in context.available_actions], dtype=float)
        probs = stable_softmax(utilities)
        return {action: float(prob) for action, prob in zip(context.available_actions, probs, strict=True)}

    def update(
        self,
        observation: Any,
        action: Any,
        outcome: Any,
        *,
        context: DecisionContext[Any],
    ) -> None:
        """Apply demonstrator-stage value-shaping and observed-outcome updates."""

        if action not in context.available_actions:
            raise ValueError(f"action {action!r} not in available_actions")

        if not is_demonstrator_stage(observation, outcome):
            return

        state = extract_state(observation)
        q_state = ensure_q_state(
            self._q_values,
            state=state,
            actions=context.available_actions,
            initial=self.initial_value,
        )

        q_state[action] += float(self.alpha_social) * (float(self.pseudo_reward) - q_state[action])

        if outcome is None:
            return

        reward = extract_reward(outcome)
        q_state[action] += float(self.alpha_observed) * (reward - q_state[action])

    def q_values_snapshot(self, *, state: int = 0) -> dict[Any, float]:
        """Return a copy of Q-values for one state."""

        return dict(self._q_values.get(int(state), {}))


@dataclass(slots=True)
class SocialObservedOutcomeValueShapingPerseverationModel:
    """Social observed-outcome value shaping model with perseveration.

    Model Contract
    --------------
    Decision Rule
        ``u[a] = beta * Q[s, a] + kappa * I[a == last_self_choice[s]]`` and
        ``P(a | s) = softmax(u[a])``.
    Update Rule
        Demonstrator-stage updates apply social shaping and observed outcome to
        demonstrator action ``demo_a``. Subject-stage updates track
        ``last_self_choice`` only.

    Parameters
    ----------
    alpha_observed : float, optional
        Learning rate for observed demonstrator outcomes in ``[0, 1]``.
    alpha_social : float, optional
        Learning rate for social value shaping in ``[0, 1]``.
    beta : float, optional
        Softmax inverse-temperature.
    kappa : float, optional
        Additive perseveration utility bonus.
    pseudo_reward : float, optional
        Pseudo-reward target for value shaping.
    initial_value : float, optional
        Initial Q-value for unseen state-action pairs.
    """

    alpha_observed: float = 0.2
    alpha_social: float = 0.2
    beta: float = 3.0
    kappa: float = 0.0
    pseudo_reward: float = 1.0
    initial_value: float = 0.0
    _q_values: dict[int, dict[Any, float]] = field(default_factory=dict, init=False, repr=False)
    _last_self_choice: dict[int, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        validate_alpha(self.alpha_observed, name="alpha_observed")
        validate_alpha(self.alpha_social, name="alpha_social")
        validate_beta(self.beta)

    def start_episode(self) -> None:
        """Reset latent values and perseveration trackers."""

        self._q_values = {}
        self._last_self_choice = {}

    def action_distribution(
        self,
        observation: Any,
        *,
        context: DecisionContext[Any],
    ) -> dict[Any, float]:
        """Return softmax action probabilities with perseveration term."""

        state = extract_state(observation)
        q_state = ensure_q_state(
            self._q_values,
            state=state,
            actions=context.available_actions,
            initial=self.initial_value,
        )
        last_choice = self._last_self_choice.get(state)
        utilities: list[float] = []
        for available_action in context.available_actions:
            stay_bonus = float(self.kappa) if available_action == last_choice else 0.0
            utilities.append(float(self.beta) * q_state[available_action] + stay_bonus)

        probs = stable_softmax(np.asarray(utilities, dtype=float))
        return {action: float(prob) for action, prob in zip(context.available_actions, probs, strict=True)}

    def update(
        self,
        observation: Any,
        action: Any,
        outcome: Any,
        *,
        context: DecisionContext[Any],
    ) -> None:
        """Track subject choice and apply demonstrator-stage value updates."""

        if action not in context.available_actions:
            raise ValueError(f"action {action!r} not in available_actions")

        state = extract_state(observation)

        if is_subject_stage(observation, outcome):
            self._last_self_choice[state] = action
            return

        if not is_demonstrator_stage(observation, outcome):
            return

        q_state = ensure_q_state(
            self._q_values,
            state=state,
            actions=context.available_actions,
            initial=self.initial_value,
        )

        q_state[action] += float(self.alpha_social) * (float(self.pseudo_reward) - q_state[action])

        if outcome is None:
            return

        reward = extract_reward(outcome)
        q_state[action] += float(self.alpha_observed) * (reward - q_state[action])

    def q_values_snapshot(self, *, state: int = 0) -> dict[Any, float]:
        """Return a copy of Q-values for one state."""

        return dict(self._q_values.get(int(state), {}))


def create_social_observed_outcome_q_model(
    *,
    alpha_observed: float = 0.2,
    beta: float = 3.0,
    initial_value: float = 0.0,
) -> SocialObservedOutcomeQModel:
    """Factory used by plugin discovery."""

    return SocialObservedOutcomeQModel(
        alpha_observed=alpha_observed,
        beta=beta,
        initial_value=initial_value,
    )


def create_social_observed_outcome_q_perseveration_model(
    *,
    alpha_observed: float = 0.2,
    beta: float = 3.0,
    kappa: float = 0.0,
    initial_value: float = 0.0,
) -> SocialObservedOutcomeQPerseverationModel:
    """Factory used by plugin discovery."""

    return SocialObservedOutcomeQPerseverationModel(
        alpha_observed=alpha_observed,
        beta=beta,
        kappa=kappa,
        initial_value=initial_value,
    )


def create_social_observed_outcome_value_shaping_model(
    *,
    alpha_observed: float = 0.2,
    alpha_social: float = 0.2,
    beta: float = 3.0,
    pseudo_reward: float = 1.0,
    initial_value: float = 0.0,
) -> SocialObservedOutcomeValueShapingModel:
    """Factory used by plugin discovery."""

    return SocialObservedOutcomeValueShapingModel(
        alpha_observed=alpha_observed,
        alpha_social=alpha_social,
        beta=beta,
        pseudo_reward=pseudo_reward,
        initial_value=initial_value,
    )


def create_social_observed_outcome_value_shaping_perseveration_model(
    *,
    alpha_observed: float = 0.2,
    alpha_social: float = 0.2,
    beta: float = 3.0,
    kappa: float = 0.0,
    pseudo_reward: float = 1.0,
    initial_value: float = 0.0,
) -> SocialObservedOutcomeValueShapingPerseverationModel:
    """Factory used by plugin discovery."""

    return SocialObservedOutcomeValueShapingPerseverationModel(
        alpha_observed=alpha_observed,
        alpha_social=alpha_social,
        beta=beta,
        kappa=kappa,
        pseudo_reward=pseudo_reward,
        initial_value=initial_value,
    )


PLUGIN_MANIFESTS = [
    ComponentManifest(
        kind="model",
        component_id="social_observed_outcome_q",
        factory=create_social_observed_outcome_q_model,
        description="Social model learning only from observed demonstrator outcomes",
        requirements=ComponentRequirements(required_outcome_fields=("reward",)),
    ),
    ComponentManifest(
        kind="model",
        component_id="social_observed_outcome_q_perseveration",
        factory=create_social_observed_outcome_q_perseveration_model,
        description="Observed-outcome social model with self perseveration",
        requirements=ComponentRequirements(required_outcome_fields=("reward",)),
    ),
    ComponentManifest(
        kind="model",
        component_id="social_observed_outcome_value_shaping",
        factory=create_social_observed_outcome_value_shaping_model,
        description="Observed-outcome social value-shaping model",
        requirements=ComponentRequirements(required_outcome_fields=("reward",)),
    ),
    ComponentManifest(
        kind="model",
        component_id="social_observed_outcome_value_shaping_perseveration",
        factory=create_social_observed_outcome_value_shaping_perseveration_model,
        description="Observed-outcome value-shaping model with self perseveration",
        requirements=ComponentRequirements(required_outcome_fields=("reward",)),
    ),
]
