"""Two-stage social bandit program with post-outcome social timing.

This program emits two decision nodes per trial in this order:

1. subject decision
2. demonstrator decision

The demonstrator node observes the subject's action/outcome from the same trial,
which represents post-outcome social information timing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from comp_model.core.contracts import DecisionContext
from comp_model.plugins import ComponentManifest
from comp_model.runtime.program import DecisionNode, TrialProgram


@dataclass(frozen=True, slots=True)
class SocialBanditPostOutcome:
    """Outcome payload for one decision node.

    Parameters
    ----------
    reward : float
        Bernoulli reward sample.
    reward_probability : float
        Bernoulli success probability for selected action.
    source_actor_id : str
        Actor whose action generated this outcome.
    """

    reward: float
    reward_probability: float
    source_actor_id: str


class TwoStageSocialPostOutcomeBanditProgram(TrialProgram):
    """Two-node social bandit with post-outcome demonstrator timing.

    Parameters
    ----------
    reward_probabilities : Sequence[float]
        Bernoulli reward probabilities for each action index.
    action_schedule : Sequence[Sequence[int]] | None, optional
        Optional per-trial action subsets.
    """

    def __init__(
        self,
        reward_probabilities: Sequence[float],
        action_schedule: Sequence[Sequence[int]] | None = None,
    ) -> None:
        if len(reward_probabilities) == 0:
            raise ValueError("reward_probabilities must contain at least one action")

        self._reward_probabilities = tuple(float(value) for value in reward_probabilities)
        for value in self._reward_probabilities:
            if value < 0.0 or value > 1.0:
                raise ValueError("reward probabilities must be within [0, 1]")

        self._all_actions = tuple(range(len(self._reward_probabilities)))

        if action_schedule is None:
            self._action_schedule = None
        else:
            normalized: list[tuple[int, ...]] = []
            for actions in action_schedule:
                trial_actions = tuple(int(action) for action in actions)
                if not trial_actions:
                    raise ValueError("each scheduled trial must expose at least one action")
                if not set(trial_actions).issubset(set(self._all_actions)):
                    raise ValueError("action_schedule includes out-of-range actions")
                normalized.append(trial_actions)
            self._action_schedule = tuple(normalized)

    def reset(self, *, rng: np.random.Generator) -> None:
        """Reset program state.

        Parameters
        ----------
        rng : numpy.random.Generator
            Runtime RNG. Unused by this stationary program.
        """

    def decision_nodes(
        self,
        *,
        trial_index: int,
        trial_events: Sequence[Any],
    ) -> tuple[DecisionNode, DecisionNode]:
        """Return subject then demonstrator decision nodes."""

        del trial_index, trial_events
        return (
            DecisionNode(node_id="subject_decision", actor_id="subject", learner_id="subject"),
            DecisionNode(node_id="demonstrator_decision", actor_id="demonstrator", learner_id="subject"),
        )

    def available_actions(
        self,
        *,
        trial_index: int,
        node: DecisionNode,
        trial_events: Sequence[Any],
    ) -> tuple[int, ...]:
        """Return legal actions for the node/trial."""

        del node, trial_events
        if self._action_schedule is None:
            return self._all_actions
        return self._action_schedule[trial_index]

    def observe(
        self,
        *,
        trial_index: int,
        node: DecisionNode,
        context: DecisionContext[Any],
        trial_events: Sequence[Any],
    ) -> dict[str, Any]:
        """Return node-specific observations.

        Demonstrator observations include subject action/outcome from the first
        decision node in the same trial.
        """

        del context
        if node.actor_id == "subject":
            return {"trial_index": trial_index, "stage": "subject"}

        subject_action = _find_payload_value(
            trial_events=trial_events,
            phase_name="decision",
            key="action",
            node_id="subject_decision",
        )
        subject_outcome = _find_payload_value(
            trial_events=trial_events,
            phase_name="outcome",
            key="outcome",
            node_id="subject_decision",
        )
        return {
            "trial_index": trial_index,
            "stage": "demonstrator",
            "subject_action": subject_action,
            "subject_outcome": subject_outcome,
        }

    def transition(
        self,
        action: Any,
        *,
        trial_index: int,
        node: DecisionNode,
        context: DecisionContext[Any],
        trial_events: Sequence[Any],
        rng: np.random.Generator,
    ) -> SocialBanditPostOutcome:
        """Apply selected action and return Bernoulli outcome."""

        del trial_index, trial_events
        if action not in context.available_actions:
            raise ValueError(f"action {action!r} is not available for node {node.node_id!r}")

        p_reward = self._reward_probabilities[int(action)]
        reward = float(rng.random() < p_reward)
        return SocialBanditPostOutcome(
            reward=reward,
            reward_probability=p_reward,
            source_actor_id=node.actor_id,
        )


def _find_payload_value(
    *,
    trial_events: Sequence[Any],
    phase_name: str,
    key: str,
    node_id: str,
) -> Any:
    """Find payload value for a specific phase/node in trial events."""

    for event in trial_events:
        if getattr(event.phase, "value", None) != phase_name:
            continue
        payload = getattr(event, "payload", None)
        if not isinstance(payload, dict):
            continue
        if payload.get("node_id") != node_id:
            continue
        if key in payload:
            return payload[key]

    raise ValueError(
        f"trial_events missing payload key {key!r} for phase {phase_name!r} and node {node_id!r}"
    )


def create_two_stage_social_post_outcome_bandit_program(
    *,
    reward_probabilities: Sequence[float],
    action_schedule: Sequence[Sequence[int]] | None = None,
) -> TwoStageSocialPostOutcomeBanditProgram:
    """Factory used by plugin discovery."""

    return TwoStageSocialPostOutcomeBanditProgram(
        reward_probabilities=reward_probabilities,
        action_schedule=action_schedule,
    )


PLUGIN_MANIFESTS = [
    ComponentManifest(
        kind="problem",
        component_id="two_stage_social_post_outcome_bandit",
        factory=create_two_stage_social_post_outcome_bandit_program,
        description="Two-phase social bandit program (subject then demonstrator)",
    )
]
