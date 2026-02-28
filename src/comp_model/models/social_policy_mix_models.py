"""Social policy-mix model family."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from comp_model.core.contracts import DecisionContext
from comp_model.core.requirements import ComponentRequirements
from comp_model.plugins import ComponentManifest

from .social_utils import (
    ensure_policy_state,
    ensure_q_state,
    maybe_extract_reward,
    extract_state,
    is_demonstrator_stage,
    is_subject_stage,
    normalized_policy_drive,
    policy_vector,
    stable_softmax,
    validate_alpha,
    validate_beta,
)


def _update_demo_policy(
    policy_state: dict[Any, float],
    *,
    actions: tuple[Any, ...],
    chosen_action: Any,
    alpha: float,
) -> None:
    """Apply RW policy update toward chosen demonstrator action."""

    for action in actions:
        target = 1.0 if action == chosen_action else 0.0
        policy_state[action] = policy_state[action] + float(alpha) * (target - policy_state[action])

    probs = policy_vector(policy_state, actions=actions)
    for action, prob in zip(actions, probs, strict=True):
        policy_state[action] = float(prob)


@dataclass(slots=True)
class SocialObservedOutcomePolicySharedMixModel:
    """Observed-outcome and policy-learning model with shared mixture weight.

    Model Contract
    --------------
    Decision Rule
        Let ``q`` be action values and ``g`` chance-normalized policy signal.
        ``drive[a] = mix_weight * q[a] + (1 - mix_weight) * g[a]``
        ``P(a | s) = softmax(beta * drive[a])``.
    Update Rule
        Demonstrator stage updates policy with ``alpha_policy`` and updates
        ``Q[s, demo_a]`` from observed demonstrator outcome with
        ``alpha_observed``. Subject stage does not update latents.

    Parameters
    ----------
    alpha_observed : float, optional
        Learning rate for observed demonstrator outcomes in ``[0, 1]``.
    alpha_policy : float, optional
        Learning rate for demonstrator policy estimate in ``[0, 1]``.
    beta : float, optional
        Shared inverse-temperature on mixed decision drive.
    mix_weight : float, optional
        Weight on value-based drive in ``[0, 1]``.
    initial_value : float, optional
        Initial Q-value for unseen state-action pairs.
    """

    alpha_observed: float = 0.2
    alpha_policy: float = 0.2
    beta: float = 6.0
    mix_weight: float = 0.5
    initial_value: float = 0.0

    _q_values: dict[int, dict[Any, float]] = field(default_factory=dict, init=False, repr=False)
    _policy_values: dict[int, dict[Any, float]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        validate_alpha(self.alpha_observed, name="alpha_observed")
        validate_alpha(self.alpha_policy, name="alpha_policy")
        validate_beta(self.beta)
        if self.mix_weight < 0.0 or self.mix_weight > 1.0:
            raise ValueError("mix_weight must be in [0, 1]")

    def start_episode(self) -> None:
        """Reset latent values and policy estimates."""

        self._q_values = {}
        self._policy_values = {}

    def action_distribution(
        self,
        observation: Any,
        *,
        context: DecisionContext[Any],
    ) -> dict[Any, float]:
        """Return softmax probabilities over mixed value-policy drive."""

        state = extract_state(observation)
        q_state = ensure_q_state(
            self._q_values,
            state=state,
            actions=context.available_actions,
            initial=self.initial_value,
        )
        policy_state = ensure_policy_state(
            self._policy_values,
            state=state,
            actions=context.available_actions,
        )

        q_vector = np.asarray([q_state[action] for action in context.available_actions], dtype=float)
        p_vector = policy_vector(policy_state, actions=context.available_actions)
        g_vector = normalized_policy_drive(p_vector)

        drive = float(self.mix_weight) * q_vector + (1.0 - float(self.mix_weight)) * g_vector
        probs = stable_softmax(float(self.beta) * drive)
        return {action: float(prob) for action, prob in zip(context.available_actions, probs, strict=True)}

    def update(
        self,
        observation: Any,
        action: Any,
        outcome: Any,
        *,
        context: DecisionContext[Any],
    ) -> None:
        """Update demonstrator policy and observed-outcome values."""

        if action not in context.available_actions:
            raise ValueError(f"action {action!r} not in available_actions")

        if not is_demonstrator_stage(observation, outcome):
            return

        state = extract_state(observation)
        policy_state = ensure_policy_state(
            self._policy_values,
            state=state,
            actions=context.available_actions,
        )
        _update_demo_policy(
            policy_state,
            actions=context.available_actions,
            chosen_action=action,
            alpha=float(self.alpha_policy),
        )

        reward = maybe_extract_reward(outcome)
        if reward is None:
            return

        q_state = ensure_q_state(
            self._q_values,
            state=state,
            actions=context.available_actions,
            initial=self.initial_value,
        )
        q_state[action] += float(self.alpha_observed) * (reward - q_state[action])

    def q_values_snapshot(self, *, state: int = 0) -> dict[Any, float]:
        """Return a copy of Q-values for one state."""

        return dict(self._q_values.get(int(state), {}))


@dataclass(slots=True)
class SocialObservedOutcomePolicySharedMixPerseverationModel:
    """Shared-mix observed-outcome and policy-learning model with perseveration.

    Model Contract
    --------------
    Decision Rule
        ``drive[a] = mix_weight * q[a] + (1 - mix_weight) * g[a]`` and
        ``u[a] = beta * drive[a] + kappa * I[a == last_self_choice[s]]``.
        ``P(a | s) = softmax(u[a])``.
    Update Rule
        Subject stage updates ``last_self_choice`` only.
        Demonstrator stage updates policy and observed-outcome values.

    Parameters
    ----------
    alpha_observed : float, optional
        Learning rate for observed demonstrator outcomes in ``[0, 1]``.
    alpha_policy : float, optional
        Learning rate for demonstrator policy estimate in ``[0, 1]``.
    beta : float, optional
        Shared inverse-temperature on mixed decision drive.
    mix_weight : float, optional
        Weight on value-based drive in ``[0, 1]``.
    kappa : float, optional
        Additive perseveration utility bonus.
    initial_value : float, optional
        Initial Q-value for unseen state-action pairs.
    """

    alpha_observed: float = 0.2
    alpha_policy: float = 0.2
    beta: float = 6.0
    mix_weight: float = 0.5
    kappa: float = 0.0
    initial_value: float = 0.0

    _q_values: dict[int, dict[Any, float]] = field(default_factory=dict, init=False, repr=False)
    _policy_values: dict[int, dict[Any, float]] = field(default_factory=dict, init=False, repr=False)
    _last_self_choice: dict[int, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        validate_alpha(self.alpha_observed, name="alpha_observed")
        validate_alpha(self.alpha_policy, name="alpha_policy")
        validate_beta(self.beta)
        if self.mix_weight < 0.0 or self.mix_weight > 1.0:
            raise ValueError("mix_weight must be in [0, 1]")

    def start_episode(self) -> None:
        """Reset latent values and policy/perseveration trackers."""

        self._q_values = {}
        self._policy_values = {}
        self._last_self_choice = {}

    def action_distribution(
        self,
        observation: Any,
        *,
        context: DecisionContext[Any],
    ) -> dict[Any, float]:
        """Return softmax probabilities over mixed drive with perseveration."""

        state = extract_state(observation)
        q_state = ensure_q_state(
            self._q_values,
            state=state,
            actions=context.available_actions,
            initial=self.initial_value,
        )
        policy_state = ensure_policy_state(
            self._policy_values,
            state=state,
            actions=context.available_actions,
        )

        q_vector = np.asarray([q_state[action] for action in context.available_actions], dtype=float)
        p_vector = policy_vector(policy_state, actions=context.available_actions)
        g_vector = normalized_policy_drive(p_vector)

        drive = float(self.mix_weight) * q_vector + (1.0 - float(self.mix_weight)) * g_vector
        utilities = float(self.beta) * drive

        last_choice = self._last_self_choice.get(state)
        for idx, action in enumerate(context.available_actions):
            if action == last_choice:
                utilities[idx] += float(self.kappa)

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
        """Update subject perseveration, demo policy, and observed-outcome Q."""

        if action not in context.available_actions:
            raise ValueError(f"action {action!r} not in available_actions")

        state = extract_state(observation)

        if is_subject_stage(observation, outcome):
            self._last_self_choice[state] = action
            return

        if not is_demonstrator_stage(observation, outcome):
            return

        policy_state = ensure_policy_state(
            self._policy_values,
            state=state,
            actions=context.available_actions,
        )
        _update_demo_policy(
            policy_state,
            actions=context.available_actions,
            chosen_action=action,
            alpha=float(self.alpha_policy),
        )

        reward = maybe_extract_reward(outcome)
        if reward is None:
            return

        q_state = ensure_q_state(
            self._q_values,
            state=state,
            actions=context.available_actions,
            initial=self.initial_value,
        )
        q_state[action] += float(self.alpha_observed) * (reward - q_state[action])

    def q_values_snapshot(self, *, state: int = 0) -> dict[Any, float]:
        """Return a copy of Q-values for one state."""

        return dict(self._q_values.get(int(state), {}))


@dataclass(slots=True)
class SocialObservedOutcomePolicyIndependentMixPerseverationModel:
    """Observed-outcome and policy model with independent decision weights.

    Model Contract
    --------------
    Decision Rule
        ``u[a] = beta_q * q[a] + beta_policy * g[a]``
        ``      + kappa * I[a == last_self_choice[s]]`` and
        ``P(a | s) = softmax(u[a])``.
    Update Rule
        Subject stage updates ``last_self_choice`` only.
        Demonstrator stage updates policy and observed-outcome values.

    Parameters
    ----------
    alpha_observed : float, optional
        Learning rate for observed demonstrator outcomes in ``[0, 1]``.
    alpha_policy : float, optional
        Learning rate for demonstrator policy estimate in ``[0, 1]``.
    beta_q : float, optional
        Decision weight on value-based drive.
    beta_policy : float, optional
        Decision weight on chance-normalized policy drive.
    kappa : float, optional
        Additive perseveration utility bonus.
    initial_value : float, optional
        Initial Q-value for unseen state-action pairs.
    """

    alpha_observed: float = 0.2
    alpha_policy: float = 0.2
    beta_q: float = 3.0
    beta_policy: float = 3.0
    kappa: float = 0.0
    initial_value: float = 0.0

    _q_values: dict[int, dict[Any, float]] = field(default_factory=dict, init=False, repr=False)
    _policy_values: dict[int, dict[Any, float]] = field(default_factory=dict, init=False, repr=False)
    _last_self_choice: dict[int, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        validate_alpha(self.alpha_observed, name="alpha_observed")
        validate_alpha(self.alpha_policy, name="alpha_policy")
        validate_beta(self.beta_q, name="beta_q")
        validate_beta(self.beta_policy, name="beta_policy")

    def start_episode(self) -> None:
        """Reset latent values and policy/perseveration trackers."""

        self._q_values = {}
        self._policy_values = {}
        self._last_self_choice = {}

    def action_distribution(
        self,
        observation: Any,
        *,
        context: DecisionContext[Any],
    ) -> dict[Any, float]:
        """Return softmax probabilities over independently weighted drives."""

        state = extract_state(observation)
        q_state = ensure_q_state(
            self._q_values,
            state=state,
            actions=context.available_actions,
            initial=self.initial_value,
        )
        policy_state = ensure_policy_state(
            self._policy_values,
            state=state,
            actions=context.available_actions,
        )

        q_vector = np.asarray([q_state[action] for action in context.available_actions], dtype=float)
        p_vector = policy_vector(policy_state, actions=context.available_actions)
        g_vector = normalized_policy_drive(p_vector)

        utilities = float(self.beta_q) * q_vector + float(self.beta_policy) * g_vector

        last_choice = self._last_self_choice.get(state)
        for idx, action in enumerate(context.available_actions):
            if action == last_choice:
                utilities[idx] += float(self.kappa)

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
        """Update subject perseveration, demo policy, and observed-outcome Q."""

        if action not in context.available_actions:
            raise ValueError(f"action {action!r} not in available_actions")

        state = extract_state(observation)

        if is_subject_stage(observation, outcome):
            self._last_self_choice[state] = action
            return

        if not is_demonstrator_stage(observation, outcome):
            return

        policy_state = ensure_policy_state(
            self._policy_values,
            state=state,
            actions=context.available_actions,
        )
        _update_demo_policy(
            policy_state,
            actions=context.available_actions,
            chosen_action=action,
            alpha=float(self.alpha_policy),
        )

        reward = maybe_extract_reward(outcome)
        if reward is None:
            return

        q_state = ensure_q_state(
            self._q_values,
            state=state,
            actions=context.available_actions,
            initial=self.initial_value,
        )
        q_state[action] += float(self.alpha_observed) * (reward - q_state[action])

    def q_values_snapshot(self, *, state: int = 0) -> dict[Any, float]:
        """Return a copy of Q-values for one state."""

        return dict(self._q_values.get(int(state), {}))


@dataclass(slots=True)
class SocialPolicyLearningOnlyModel:
    """Social policy-learning-only model (no value learning).

    Model Contract
    --------------
    Decision Rule
        Let ``g`` be chance-normalized demonstrator policy signal.
        ``P(a | s) = softmax(beta * g[a])``.
    Update Rule
        Demonstrator stage updates policy estimate with ``alpha_policy``.
        Subject stage has no latent update.

    Parameters
    ----------
    alpha_policy : float, optional
        Learning rate for demonstrator policy estimate in ``[0, 1]``.
    beta : float, optional
        Inverse-temperature on policy drive.
    """

    alpha_policy: float = 0.2
    beta: float = 6.0

    _policy_values: dict[int, dict[Any, float]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        validate_alpha(self.alpha_policy, name="alpha_policy")
        validate_beta(self.beta)

    def start_episode(self) -> None:
        """Reset policy estimates."""

        self._policy_values = {}

    def action_distribution(
        self,
        observation: Any,
        *,
        context: DecisionContext[Any],
    ) -> dict[Any, float]:
        """Return softmax probabilities over chance-normalized policy drive."""

        state = extract_state(observation)
        policy_state = ensure_policy_state(
            self._policy_values,
            state=state,
            actions=context.available_actions,
        )
        p_vector = policy_vector(policy_state, actions=context.available_actions)
        g_vector = normalized_policy_drive(p_vector)

        probs = stable_softmax(float(self.beta) * g_vector)
        return {action: float(prob) for action, prob in zip(context.available_actions, probs, strict=True)}

    def update(
        self,
        observation: Any,
        action: Any,
        outcome: Any,
        *,
        context: DecisionContext[Any],
    ) -> None:
        """Update demonstrator policy estimate only."""

        if action not in context.available_actions:
            raise ValueError(f"action {action!r} not in available_actions")

        if not is_demonstrator_stage(observation, outcome):
            return

        state = extract_state(observation)
        policy_state = ensure_policy_state(
            self._policy_values,
            state=state,
            actions=context.available_actions,
        )
        _update_demo_policy(
            policy_state,
            actions=context.available_actions,
            chosen_action=action,
            alpha=float(self.alpha_policy),
        )


@dataclass(slots=True)
class SocialPolicyLearningOnlyPerseverationModel:
    """Policy-learning-only social model with self perseveration.

    Model Contract
    --------------
    Decision Rule
        ``u[a] = beta * g[a] + kappa * I[a == last_self_choice[s]]`` and
        ``P(a | s) = softmax(u[a])``.
    Update Rule
        Subject stage updates ``last_self_choice`` only.
        Demonstrator stage updates policy estimate with ``alpha_policy``.

    Parameters
    ----------
    alpha_policy : float, optional
        Learning rate for demonstrator policy estimate in ``[0, 1]``.
    beta : float, optional
        Inverse-temperature on policy drive.
    kappa : float, optional
        Additive perseveration utility bonus.
    """

    alpha_policy: float = 0.2
    beta: float = 6.0
    kappa: float = 0.0

    _policy_values: dict[int, dict[Any, float]] = field(default_factory=dict, init=False, repr=False)
    _last_self_choice: dict[int, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        validate_alpha(self.alpha_policy, name="alpha_policy")
        validate_beta(self.beta)

    def start_episode(self) -> None:
        """Reset policy estimates and perseveration trackers."""

        self._policy_values = {}
        self._last_self_choice = {}

    def action_distribution(
        self,
        observation: Any,
        *,
        context: DecisionContext[Any],
    ) -> dict[Any, float]:
        """Return softmax probabilities over policy drive with perseveration."""

        state = extract_state(observation)
        policy_state = ensure_policy_state(
            self._policy_values,
            state=state,
            actions=context.available_actions,
        )
        p_vector = policy_vector(policy_state, actions=context.available_actions)
        g_vector = normalized_policy_drive(p_vector)

        utilities = float(self.beta) * g_vector
        last_choice = self._last_self_choice.get(state)
        for idx, action in enumerate(context.available_actions):
            if action == last_choice:
                utilities[idx] += float(self.kappa)

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
        """Update subject perseveration and demonstrator policy estimate."""

        if action not in context.available_actions:
            raise ValueError(f"action {action!r} not in available_actions")

        state = extract_state(observation)

        if is_subject_stage(observation, outcome):
            self._last_self_choice[state] = action
            return

        if not is_demonstrator_stage(observation, outcome):
            return

        policy_state = ensure_policy_state(
            self._policy_values,
            state=state,
            actions=context.available_actions,
        )
        _update_demo_policy(
            policy_state,
            actions=context.available_actions,
            chosen_action=action,
            alpha=float(self.alpha_policy),
        )


def create_social_observed_outcome_policy_shared_mix_model(
    *,
    alpha_observed: float = 0.2,
    alpha_policy: float = 0.2,
    beta: float = 6.0,
    mix_weight: float = 0.5,
    initial_value: float = 0.0,
) -> SocialObservedOutcomePolicySharedMixModel:
    """Factory used by plugin discovery."""

    return SocialObservedOutcomePolicySharedMixModel(
        alpha_observed=alpha_observed,
        alpha_policy=alpha_policy,
        beta=beta,
        mix_weight=mix_weight,
        initial_value=initial_value,
    )


def create_social_observed_outcome_policy_shared_mix_perseveration_model(
    *,
    alpha_observed: float = 0.2,
    alpha_policy: float = 0.2,
    beta: float = 6.0,
    mix_weight: float = 0.5,
    kappa: float = 0.0,
    initial_value: float = 0.0,
) -> SocialObservedOutcomePolicySharedMixPerseverationModel:
    """Factory used by plugin discovery."""

    return SocialObservedOutcomePolicySharedMixPerseverationModel(
        alpha_observed=alpha_observed,
        alpha_policy=alpha_policy,
        beta=beta,
        mix_weight=mix_weight,
        kappa=kappa,
        initial_value=initial_value,
    )


def create_social_observed_outcome_policy_independent_mix_perseveration_model(
    *,
    alpha_observed: float = 0.2,
    alpha_policy: float = 0.2,
    beta_q: float = 3.0,
    beta_policy: float = 3.0,
    kappa: float = 0.0,
    initial_value: float = 0.0,
) -> SocialObservedOutcomePolicyIndependentMixPerseverationModel:
    """Factory used by plugin discovery."""

    return SocialObservedOutcomePolicyIndependentMixPerseverationModel(
        alpha_observed=alpha_observed,
        alpha_policy=alpha_policy,
        beta_q=beta_q,
        beta_policy=beta_policy,
        kappa=kappa,
        initial_value=initial_value,
    )


def create_social_policy_learning_only_model(
    *,
    alpha_policy: float = 0.2,
    beta: float = 6.0,
) -> SocialPolicyLearningOnlyModel:
    """Factory used by plugin discovery."""

    return SocialPolicyLearningOnlyModel(
        alpha_policy=alpha_policy,
        beta=beta,
    )


def create_social_policy_learning_only_perseveration_model(
    *,
    alpha_policy: float = 0.2,
    beta: float = 6.0,
    kappa: float = 0.0,
) -> SocialPolicyLearningOnlyPerseverationModel:
    """Factory used by plugin discovery."""

    return SocialPolicyLearningOnlyPerseverationModel(
        alpha_policy=alpha_policy,
        beta=beta,
        kappa=kappa,
    )


PLUGIN_MANIFESTS = [
    ComponentManifest(
        kind="model",
        component_id="social_observed_outcome_policy_shared_mix",
        factory=create_social_observed_outcome_policy_shared_mix_model,
        description="Observed-outcome + policy model with shared mixture drive",
        requirements=ComponentRequirements(required_outcome_fields=("reward",)),
    ),
    ComponentManifest(
        kind="model",
        component_id="social_observed_outcome_policy_shared_mix_perseveration",
        factory=create_social_observed_outcome_policy_shared_mix_perseveration_model,
        description="Shared-mix observed-outcome + policy model with perseveration",
        requirements=ComponentRequirements(required_outcome_fields=("reward",)),
    ),
    ComponentManifest(
        kind="model",
        component_id="social_observed_outcome_policy_independent_mix_perseveration",
        factory=create_social_observed_outcome_policy_independent_mix_perseveration_model,
        description="Independent-weight observed-outcome + policy model with perseveration",
        requirements=ComponentRequirements(required_outcome_fields=("reward",)),
    ),
    ComponentManifest(
        kind="model",
        component_id="social_policy_learning_only",
        factory=create_social_policy_learning_only_model,
        description="Policy-learning-only social model",
    ),
    ComponentManifest(
        kind="model",
        component_id="social_policy_learning_only_perseveration",
        factory=create_social_policy_learning_only_perseveration_model,
        description="Policy-learning-only social model with perseveration",
    ),
]
