"""Social reliability-gated and demo-bias model family."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from comp_model.core.contracts import DecisionContext
from comp_model.core.requirements import ComponentRequirements
from comp_model.plugins import ComponentManifest

from .social_utils import (
    count_vector,
    determinism_index_reliability,
    ensure_count_state,
    ensure_policy_state,
    ensure_q_state,
    maybe_extract_reward,
    extract_state,
    is_demonstrator_stage,
    is_subject_stage,
    policy_vector,
    stable_softmax,
    validate_alpha,
    validate_beta,
)


def _update_policy_with_action(
    policy_state: dict[Any, float],
    *,
    actions: tuple[Any, ...],
    chosen_action: Any,
    alpha: float,
) -> None:
    """Apply RW policy update toward one-hot on chosen action."""

    for action in actions:
        target = 1.0 if action == chosen_action else 0.0
        policy_state[action] = policy_state[action] + float(alpha) * (target - policy_state[action])

    probs = policy_vector(policy_state, actions=actions)
    for action, prob in zip(actions, probs, strict=True):
        policy_state[action] = float(prob)


@dataclass(slots=True)
class SocialPolicyReliabilityGatedValueShapingModel:
    """Social model with policy-reliability-gated value shaping.

    Model Contract
    --------------
    Decision Rule
        ``u[a] = beta * Q[s, a] + kappa * I[a == last_self_choice[s]]`` and
        ``P(a | s) = softmax(u[a])``.
    Update Rule
        Subject stage: update ``last_self_choice`` only.

        Demonstrator stage:
        1. Update demonstrator policy estimate with ``alpha_policy``.
        2. Compute reliability ``Rel`` from policy determinism.
        3. Apply value shaping with effective rate ``alpha_social_base * Rel``.
        4. Apply observed-outcome update with ``alpha_observed`` when reward is observed.

    Parameters
    ----------
    alpha_observed : float, optional
        Learning rate for observed demonstrator outcomes in ``[0, 1]``.
    alpha_social_base : float, optional
        Base social shaping rate in ``[0, 1]``.
    alpha_policy : float, optional
        Learning rate for demonstrator policy estimate in ``[0, 1]``.
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
    alpha_social_base: float = 0.2
    alpha_policy: float = 0.2
    beta: float = 3.0
    kappa: float = 0.0
    pseudo_reward: float = 1.0
    initial_value: float = 0.0

    _q_values: dict[int, dict[Any, float]] = field(default_factory=dict, init=False, repr=False)
    _policy_values: dict[int, dict[Any, float]] = field(default_factory=dict, init=False, repr=False)
    _last_self_choice: dict[int, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        validate_alpha(self.alpha_observed, name="alpha_observed")
        validate_alpha(self.alpha_social_base, name="alpha_social_base")
        validate_alpha(self.alpha_policy, name="alpha_policy")
        validate_beta(self.beta)

    def start_episode(self) -> None:
        """Reset latent values and internal trackers."""

        self._q_values = {}
        self._policy_values = {}
        self._last_self_choice = {}

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
        last_choice = self._last_self_choice.get(state)
        utilities: list[float] = []
        for action in context.available_actions:
            stay_bonus = float(self.kappa) if action == last_choice else 0.0
            utilities.append(float(self.beta) * q_state[action] + stay_bonus)

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
        """Apply subject perseveration update and demonstrator social learning."""

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
        policy_state = ensure_policy_state(
            self._policy_values,
            state=state,
            actions=context.available_actions,
        )

        _update_policy_with_action(
            policy_state,
            actions=context.available_actions,
            chosen_action=action,
            alpha=float(self.alpha_policy),
        )
        rel = determinism_index_reliability(policy_vector(policy_state, actions=context.available_actions))
        alpha_social = float(self.alpha_social_base) * rel
        q_state[action] += alpha_social * (float(self.pseudo_reward) - q_state[action])

        reward = maybe_extract_reward(outcome)
        if reward is None:
            return

        q_state[action] += float(self.alpha_observed) * (reward - q_state[action])

    def q_values_snapshot(self, *, state: int = 0) -> dict[Any, float]:
        """Return a copy of Q-values for one state."""

        return dict(self._q_values.get(int(state), {}))


@dataclass(slots=True)
class SocialConstantDemoBiasObservedOutcomeQPerseverationModel:
    """Observed-outcome social Q model with constant demo-action bias.

    Model Contract
    --------------
    Decision Rule
        ``u[a] = beta * Q[s, a] + kappa * I[a == last_self_choice[s]]``
        ``      + demo_bias * I[a == recent_demo_choice[s]]``
        and ``P(a | s) = softmax(u[a])``.
    Update Rule
        Subject stage updates ``last_self_choice`` only.
        Demonstrator stage sets ``recent_demo_choice`` and updates
        ``Q[s, demo_a]`` from observed outcome with ``alpha_observed``.

    Parameters
    ----------
    alpha_observed : float, optional
        Learning rate for observed demonstrator outcomes in ``[0, 1]``.
    demo_bias : float, optional
        Constant additive utility bonus for copying recent demo action.
    beta : float, optional
        Softmax inverse-temperature.
    kappa : float, optional
        Additive perseveration utility bonus.
    initial_value : float, optional
        Initial Q-value for unseen state-action pairs.
    """

    alpha_observed: float = 0.2
    demo_bias: float = 1.0
    beta: float = 3.0
    kappa: float = 0.0
    initial_value: float = 0.0

    _q_values: dict[int, dict[Any, float]] = field(default_factory=dict, init=False, repr=False)
    _last_self_choice: dict[int, Any] = field(default_factory=dict, init=False, repr=False)
    _recent_demo_choice: dict[int, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        validate_alpha(self.alpha_observed, name="alpha_observed")
        validate_beta(self.beta)

    def start_episode(self) -> None:
        """Reset latent values and choice trackers."""

        self._q_values = {}
        self._last_self_choice = {}
        self._recent_demo_choice = {}

    def action_distribution(
        self,
        observation: Any,
        *,
        context: DecisionContext[Any],
    ) -> dict[Any, float]:
        """Return softmax action probabilities with copy and stay biases."""

        state = extract_state(observation)
        q_state = ensure_q_state(
            self._q_values,
            state=state,
            actions=context.available_actions,
            initial=self.initial_value,
        )
        last_choice = self._last_self_choice.get(state)
        demo_choice = self._recent_demo_choice.get(state)

        utilities: list[float] = []
        for action in context.available_actions:
            stay_bonus = float(self.kappa) if action == last_choice else 0.0
            demo_bonus = float(self.demo_bias) if action == demo_choice else 0.0
            utilities.append(float(self.beta) * q_state[action] + stay_bonus + demo_bonus)

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
        """Apply subject perseveration update and demo-stage outcome learning."""

        if action not in context.available_actions:
            raise ValueError(f"action {action!r} not in available_actions")

        state = extract_state(observation)

        if is_subject_stage(observation, outcome):
            self._last_self_choice[state] = action
            return

        if not is_demonstrator_stage(observation, outcome):
            return

        self._recent_demo_choice[state] = action

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
class SocialPolicyReliabilityGatedDemoBiasObservedOutcomeQPerseverationModel:
    """Observed-outcome social Q model with reliability-gated demo bias.

    Model Contract
    --------------
    Decision Rule
        Let ``Rel[s]`` be determinism reliability of demonstrator policy estimate.
        ``demo_bias = demo_bias_rel * Rel[s]``
        ``u[a] = beta * Q[s, a] + kappa * I[a == last_self_choice[s]]``
        ``      + demo_bias * I[a == recent_demo_choice[s]]``
        and ``P(a | s) = softmax(u[a])``.
    Update Rule
        Subject stage updates ``last_self_choice`` only.

        Demonstrator stage:
        1. Update demonstrator policy estimate with ``alpha_policy``.
        2. Set ``recent_demo_choice``.
        3. Update observed-outcome Q-value with ``alpha_observed`` when reward exists.

    Parameters
    ----------
    alpha_observed : float, optional
        Learning rate for observed demonstrator outcomes in ``[0, 1]``.
    alpha_policy : float, optional
        Learning rate for demonstrator policy estimate in ``[0, 1]``.
    demo_bias_rel : float, optional
        Reliability-scaled copying bias weight.
    beta : float, optional
        Softmax inverse-temperature.
    kappa : float, optional
        Additive perseveration utility bonus.
    initial_value : float, optional
        Initial Q-value for unseen state-action pairs.
    """

    alpha_observed: float = 0.2
    alpha_policy: float = 0.2
    demo_bias_rel: float = 1.0
    beta: float = 3.0
    kappa: float = 0.0
    initial_value: float = 0.0

    _q_values: dict[int, dict[Any, float]] = field(default_factory=dict, init=False, repr=False)
    _policy_values: dict[int, dict[Any, float]] = field(default_factory=dict, init=False, repr=False)
    _last_self_choice: dict[int, Any] = field(default_factory=dict, init=False, repr=False)
    _recent_demo_choice: dict[int, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        validate_alpha(self.alpha_observed, name="alpha_observed")
        validate_alpha(self.alpha_policy, name="alpha_policy")
        validate_beta(self.beta)

    def start_episode(self) -> None:
        """Reset latent values and internal trackers."""

        self._q_values = {}
        self._policy_values = {}
        self._last_self_choice = {}
        self._recent_demo_choice = {}

    def action_distribution(
        self,
        observation: Any,
        *,
        context: DecisionContext[Any],
    ) -> dict[Any, float]:
        """Return softmax action probabilities with reliability-gated demo bias."""

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
        rel = determinism_index_reliability(policy_vector(policy_state, actions=context.available_actions))
        demo_bias = float(self.demo_bias_rel) * rel

        last_choice = self._last_self_choice.get(state)
        demo_choice = self._recent_demo_choice.get(state)
        utilities: list[float] = []
        for action in context.available_actions:
            stay_bonus = float(self.kappa) if action == last_choice else 0.0
            demo_bonus = demo_bias if action == demo_choice else 0.0
            utilities.append(float(self.beta) * q_state[action] + stay_bonus + demo_bonus)

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
        """Update subject perseveration, demo policy, and observed-outcome Q."""

        if action not in context.available_actions:
            raise ValueError(f"action {action!r} not in available_actions")

        state = extract_state(observation)

        if is_subject_stage(observation, outcome):
            self._last_self_choice[state] = action
            return

        if not is_demonstrator_stage(observation, outcome):
            return

        self._recent_demo_choice[state] = action
        policy_state = ensure_policy_state(
            self._policy_values,
            state=state,
            actions=context.available_actions,
        )
        _update_policy_with_action(
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
class SocialDirichletReliabilityGatedDemoBiasObservedOutcomeQPerseverationModel:
    """Observed-outcome social Q model with Dirichlet reliability-gated demo bias.

    Model Contract
    --------------
    Decision Rule
        Reliability is computed from Dirichlet-count policy estimate.
        ``demo_bias = demo_bias_rel * Rel[s]``
        ``u[a] = beta * Q[s, a] + kappa * I[a == last_self_choice[s]]``
        ``      + demo_bias * I[a == recent_demo_choice[s]]``
        and ``P(a | s) = softmax(u[a])``.
    Update Rule
        Subject stage updates ``last_self_choice`` only.

        Demonstrator stage:
        1. Increment action count for observed demonstrator action.
        2. Set ``recent_demo_choice``.
        3. Update observed-outcome Q-value with ``alpha_observed`` when reward exists.

    Parameters
    ----------
    alpha_observed : float, optional
        Learning rate for observed demonstrator outcomes in ``[0, 1]``.
    demo_bias_rel : float, optional
        Reliability-scaled copying bias weight.
    beta : float, optional
        Softmax inverse-temperature.
    kappa : float, optional
        Additive perseveration utility bonus.
    demo_dirichlet_prior : float, optional
        Symmetric prior pseudocount per action (must be > 0).
    initial_value : float, optional
        Initial Q-value for unseen state-action pairs.
    """

    alpha_observed: float = 0.2
    demo_bias_rel: float = 1.0
    beta: float = 3.0
    kappa: float = 0.0
    demo_dirichlet_prior: float = 1.0
    initial_value: float = 0.0

    _q_values: dict[int, dict[Any, float]] = field(default_factory=dict, init=False, repr=False)
    _count_values: dict[int, dict[Any, float]] = field(default_factory=dict, init=False, repr=False)
    _last_self_choice: dict[int, Any] = field(default_factory=dict, init=False, repr=False)
    _recent_demo_choice: dict[int, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        validate_alpha(self.alpha_observed, name="alpha_observed")
        validate_beta(self.beta)
        if self.demo_dirichlet_prior <= 0.0:
            raise ValueError("demo_dirichlet_prior must be > 0")

    def start_episode(self) -> None:
        """Reset latent values and internal trackers."""

        self._q_values = {}
        self._count_values = {}
        self._last_self_choice = {}
        self._recent_demo_choice = {}

    def action_distribution(
        self,
        observation: Any,
        *,
        context: DecisionContext[Any],
    ) -> dict[Any, float]:
        """Return softmax action probabilities with Dirichlet-gated demo bias."""

        state = extract_state(observation)
        q_state = ensure_q_state(
            self._q_values,
            state=state,
            actions=context.available_actions,
            initial=self.initial_value,
        )
        count_state = ensure_count_state(
            self._count_values,
            state=state,
            actions=context.available_actions,
            prior=float(self.demo_dirichlet_prior),
        )

        rel = determinism_index_reliability(count_vector(count_state, actions=context.available_actions))
        demo_bias = float(self.demo_bias_rel) * rel

        last_choice = self._last_self_choice.get(state)
        demo_choice = self._recent_demo_choice.get(state)

        utilities: list[float] = []
        for action in context.available_actions:
            stay_bonus = float(self.kappa) if action == last_choice else 0.0
            demo_bonus = demo_bias if action == demo_choice else 0.0
            utilities.append(float(self.beta) * q_state[action] + stay_bonus + demo_bonus)

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
        """Update subject perseveration, demo counts, and observed-outcome Q."""

        if action not in context.available_actions:
            raise ValueError(f"action {action!r} not in available_actions")

        state = extract_state(observation)

        if is_subject_stage(observation, outcome):
            self._last_self_choice[state] = action
            return

        if not is_demonstrator_stage(observation, outcome):
            return

        self._recent_demo_choice[state] = action
        count_state = ensure_count_state(
            self._count_values,
            state=state,
            actions=context.available_actions,
            prior=float(self.demo_dirichlet_prior),
        )
        count_state[action] = float(count_state[action]) + 1.0

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


def create_social_policy_reliability_gated_value_shaping_model(
    *,
    alpha_observed: float = 0.2,
    alpha_social_base: float = 0.2,
    alpha_policy: float = 0.2,
    beta: float = 3.0,
    kappa: float = 0.0,
    pseudo_reward: float = 1.0,
    initial_value: float = 0.0,
) -> SocialPolicyReliabilityGatedValueShapingModel:
    """Factory used by plugin discovery."""

    return SocialPolicyReliabilityGatedValueShapingModel(
        alpha_observed=alpha_observed,
        alpha_social_base=alpha_social_base,
        alpha_policy=alpha_policy,
        beta=beta,
        kappa=kappa,
        pseudo_reward=pseudo_reward,
        initial_value=initial_value,
    )


def create_social_constant_demo_bias_observed_outcome_q_perseveration_model(
    *,
    alpha_observed: float = 0.2,
    demo_bias: float = 1.0,
    beta: float = 3.0,
    kappa: float = 0.0,
    initial_value: float = 0.0,
) -> SocialConstantDemoBiasObservedOutcomeQPerseverationModel:
    """Factory used by plugin discovery."""

    return SocialConstantDemoBiasObservedOutcomeQPerseverationModel(
        alpha_observed=alpha_observed,
        demo_bias=demo_bias,
        beta=beta,
        kappa=kappa,
        initial_value=initial_value,
    )


def create_social_policy_reliability_gated_demo_bias_observed_outcome_q_perseveration_model(
    *,
    alpha_observed: float = 0.2,
    alpha_policy: float = 0.2,
    demo_bias_rel: float = 1.0,
    beta: float = 3.0,
    kappa: float = 0.0,
    initial_value: float = 0.0,
) -> SocialPolicyReliabilityGatedDemoBiasObservedOutcomeQPerseverationModel:
    """Factory used by plugin discovery."""

    return SocialPolicyReliabilityGatedDemoBiasObservedOutcomeQPerseverationModel(
        alpha_observed=alpha_observed,
        alpha_policy=alpha_policy,
        demo_bias_rel=demo_bias_rel,
        beta=beta,
        kappa=kappa,
        initial_value=initial_value,
    )


def create_social_dirichlet_reliability_gated_demo_bias_observed_outcome_q_perseveration_model(
    *,
    alpha_observed: float = 0.2,
    demo_bias_rel: float = 1.0,
    beta: float = 3.0,
    kappa: float = 0.0,
    demo_dirichlet_prior: float = 1.0,
    initial_value: float = 0.0,
) -> SocialDirichletReliabilityGatedDemoBiasObservedOutcomeQPerseverationModel:
    """Factory used by plugin discovery."""

    return SocialDirichletReliabilityGatedDemoBiasObservedOutcomeQPerseverationModel(
        alpha_observed=alpha_observed,
        demo_bias_rel=demo_bias_rel,
        beta=beta,
        kappa=kappa,
        demo_dirichlet_prior=demo_dirichlet_prior,
        initial_value=initial_value,
    )


PLUGIN_MANIFESTS = [
    ComponentManifest(
        kind="model",
        component_id="social_policy_reliability_gated_value_shaping",
        factory=create_social_policy_reliability_gated_value_shaping_model,
        description="Policy-reliability-gated social value shaping with observed outcomes",
        requirements=ComponentRequirements(required_outcome_fields=("reward",)),
    ),
    ComponentManifest(
        kind="model",
        component_id="social_constant_demo_bias_observed_outcome_q_perseveration",
        factory=create_social_constant_demo_bias_observed_outcome_q_perseveration_model,
        description="Observed-outcome Q model with constant demo-copy bias and perseveration",
        requirements=ComponentRequirements(required_outcome_fields=("reward",)),
    ),
    ComponentManifest(
        kind="model",
        component_id="social_policy_reliability_gated_demo_bias_observed_outcome_q_perseveration",
        factory=create_social_policy_reliability_gated_demo_bias_observed_outcome_q_perseveration_model,
        description="Observed-outcome Q model with reliability-gated demo-copy bias and perseveration",
        requirements=ComponentRequirements(required_outcome_fields=("reward",)),
    ),
    ComponentManifest(
        kind="model",
        component_id="social_dirichlet_reliability_gated_demo_bias_observed_outcome_q_perseveration",
        factory=create_social_dirichlet_reliability_gated_demo_bias_observed_outcome_q_perseveration_model,
        description="Observed-outcome Q model with Dirichlet reliability-gated demo-copy bias and perseveration",
        requirements=ComponentRequirements(required_outcome_fields=("reward",)),
    ),
]
