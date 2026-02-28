"""Social value-shaping model with self-outcome learning.

This model corresponds to the classic value-shaping family where observed
partner actions provide a pseudo-reward teaching signal and self outcomes drive
chosen-action reinforcement learning.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from comp_model.core.contracts import DecisionContext
from comp_model.core.requirements import ComponentRequirements
from comp_model.plugins import ComponentManifest


@dataclass(slots=True)
class SocialSelfOutcomeValueShapingModel:
    """Social value-shaping model with self-outcome learning.

    Model Contract
    --------------
    Decision Rule
        Let ``Q[s, a]`` be action values, ``beta`` inverse temperature, and
        ``kappa`` perseveration strength for repeating previous self action:
        ``u[a] = beta * Q[s, a] + kappa * I[a == last_self_choice[s]]`` and
        ``P(a | s) = softmax(u[a])``.

        If observation includes ``demonstrator_action`` for the current trial,
        value shaping is applied before probability computation so social
        information can influence the same trial's decision.
    Update Rule
        Social shaping (pre-decision):
        ``Q[s, demo_a] <- Q[s, demo_a] + alpha_social * (pseudo_reward - Q[s, demo_a])``.

        Self-outcome update (post-outcome, subject stage only):
        ``Q[s, a] <- Q[s, a] + alpha_self * (r - Q[s, a])``.

        Demonstrator-stage updates are ignored for self-outcome learning.

    Parameters
    ----------
    alpha_self : float, optional
        Learning rate for self outcome updates in ``[0, 1]``.
    alpha_social : float, optional
        Learning rate for social value shaping in ``[0, 1]``.
    beta : float, optional
        Softmax inverse-temperature.
    kappa : float, optional
        Additive perseveration bonus for repeating previous self action.
    pseudo_reward : float, optional
        Pseudo-reward target used for social value shaping.
    initial_value : float, optional
        Initial Q-value for unseen state-action pairs.
    """

    alpha_self: float = 0.2
    alpha_social: float = 0.2
    beta: float = 3.0
    kappa: float = 0.0
    pseudo_reward: float = 1.0
    initial_value: float = 0.0

    _q_values: dict[int, dict[Any, float]] = field(default_factory=dict, init=False, repr=False)
    _last_self_choice: dict[int, Any] = field(default_factory=dict, init=False, repr=False)
    _social_applied: set[tuple[int, int, int]] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_alpha(self.alpha_self, name="alpha_self")
        _validate_alpha(self.alpha_social, name="alpha_social")
        _validate_beta(self.beta)

    def start_episode(self) -> None:
        """Reset latent values and per-episode bookkeeping."""

        self._q_values = {}
        self._last_self_choice = {}
        self._social_applied = set()

    def action_distribution(
        self,
        observation: Any,
        *,
        context: DecisionContext[Any],
    ) -> dict[Any, float]:
        """Return action probabilities with optional social value shaping.

        Parameters
        ----------
        observation : Any
            Observation payload. When mapping includes ``demonstrator_action``,
            the model applies one social shaping update before computing
            probabilities.
        context : DecisionContext[Any]
            Per-trial context containing legal actions.

        Returns
        -------
        dict[Any, float]
            Action probability distribution.
        """

        state = _extract_state(observation)
        q_state = _ensure_q_state(self._q_values, state=state, actions=context.available_actions, initial=self.initial_value)

        # Apply social shaping at decision time once per trial/node.
        self._apply_social_shaping(observation=observation, context=context, state=state, q_state=q_state)

        last_choice = self._last_self_choice.get(state)
        utilities: list[float] = []
        for action in context.available_actions:
            stay_bonus = float(self.kappa) if action == last_choice else 0.0
            utilities.append(float(self.beta) * q_state[action] + stay_bonus)

        probs = _softmax(np.asarray(utilities, dtype=float))
        return {action: float(prob) for action, prob in zip(context.available_actions, probs, strict=True)}

    def update(
        self,
        observation: Any,
        action: Any,
        outcome: Any,
        *,
        context: DecisionContext[Any],
    ) -> None:
        """Apply self-outcome learning for subject stage decisions.

        Parameters
        ----------
        observation : Any
            Observation payload.
        action : Any
            Selected action.
        outcome : Any
            Outcome payload.
        context : DecisionContext[Any]
            Per-trial context.
        """

        if action not in context.available_actions:
            raise ValueError(f"action {action!r} not in available_actions")

        # In two-stage social programs, demonstrator node updates target the
        # subject learner but should not be treated as subject private learning.
        if not _is_subject_stage(observation):
            return

        state = _extract_state(observation)
        q_state = _ensure_q_state(self._q_values, state=state, actions=context.available_actions, initial=self.initial_value)
        self._last_self_choice[state] = action

        if outcome is None:
            return

        reward = _extract_reward(outcome)
        q_state[action] += float(self.alpha_self) * (reward - q_state[action])

    def q_values_snapshot(self, *, state: int = 0) -> dict[Any, float]:
        """Return a copy of Q-values for one state."""

        return dict(self._q_values.get(int(state), {}))

    def _apply_social_shaping(
        self,
        *,
        observation: Any,
        context: DecisionContext[Any],
        state: int,
        q_state: dict[Any, float],
    ) -> None:
        """Apply one social shaping step if demonstrator action is observed."""

        if not isinstance(observation, Mapping):
            return
        if "demonstrator_action" not in observation:
            return

        key = (int(context.trial_index), int(context.decision_index), int(state))
        if key in self._social_applied:
            return

        demo_action = observation["demonstrator_action"]
        if demo_action not in context.available_actions:
            raise ValueError("demonstrator_action is not in available_actions")

        q_state[demo_action] += float(self.alpha_social) * (float(self.pseudo_reward) - q_state[demo_action])
        self._social_applied.add(key)


def _extract_state(observation: Any) -> int:
    """Extract latent state index from observation payload."""

    if isinstance(observation, Mapping) and "state" in observation:
        return int(observation["state"])
    return 0


def _is_subject_stage(observation: Any) -> bool:
    """Return whether an observation corresponds to subject decision stage."""

    if not isinstance(observation, Mapping):
        return True

    stage = observation.get("stage")
    if stage is None:
        return True

    return str(stage) == "subject"


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


def _softmax(values: np.ndarray) -> np.ndarray:
    """Compute numerically stable softmax probabilities."""

    if values.size == 1:
        return np.asarray([1.0], dtype=float)

    logits = np.asarray(values, dtype=float)
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


def create_social_self_outcome_value_shaping_model(
    *,
    alpha_self: float = 0.2,
    alpha_social: float = 0.2,
    beta: float = 3.0,
    kappa: float = 0.0,
    pseudo_reward: float = 1.0,
    initial_value: float = 0.0,
) -> SocialSelfOutcomeValueShapingModel:
    """Factory used by plugin discovery."""

    return SocialSelfOutcomeValueShapingModel(
        alpha_self=alpha_self,
        alpha_social=alpha_social,
        beta=beta,
        kappa=kappa,
        pseudo_reward=pseudo_reward,
        initial_value=initial_value,
    )


PLUGIN_MANIFESTS = [
    ComponentManifest(
        kind="model",
        component_id="social_self_outcome_value_shaping",
        factory=create_social_self_outcome_value_shaping_model,
        description="Social value shaping with self-outcome learning and perseveration",
        requirements=ComponentRequirements(required_outcome_fields=("reward",)),
    )
]
