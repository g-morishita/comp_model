"""Ordered two-actor social bandit programs.

These programs model social-learning timing variants by emitting explicit
observation/decision/outcome/update steps in program-defined order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from comp_model.core.contracts import DecisionContext
from comp_model.core.events import EventPhase
from comp_model.plugins import ComponentManifest
from comp_model.runtime.program import ProgramStep, TrialProgram


@dataclass(frozen=True, slots=True)
class TwoActorSocialBanditOutcome:
    """Outcome payload for one two-actor social bandit decision."""

    reward: float
    reward_probability: float
    source_actor_id: str


class _TwoActorOrderedSocialBanditProgram(TrialProgram):
    """Internal configurable two-actor social-learning program."""

    def __init__(
        self,
        reward_probabilities: Sequence[float],
        *,
        first_actor_id: str,
        second_actor_id: str,
        first_outcome_observed: bool,
        second_self_outcome: bool,
        action_schedule: Sequence[Sequence[int]] | None = None,
    ) -> None:
        if len(reward_probabilities) == 0:
            raise ValueError("reward_probabilities must contain at least one action")
        if first_actor_id.strip() == "" or second_actor_id.strip() == "":
            raise ValueError("actor IDs must be non-empty strings")
        if first_actor_id == second_actor_id:
            raise ValueError("first_actor_id and second_actor_id must differ")

        self._reward_probabilities = tuple(float(value) for value in reward_probabilities)
        for value in self._reward_probabilities:
            if value < 0.0 or value > 1.0:
                raise ValueError("reward probabilities must be within [0, 1]")

        self._all_actions = tuple(range(len(self._reward_probabilities)))
        self._action_schedule = _normalize_action_schedule(action_schedule, all_actions=self._all_actions)
        self._first_actor_id = first_actor_id
        self._second_actor_id = second_actor_id
        self._first_outcome_observed = bool(first_outcome_observed)
        self._second_self_outcome = bool(second_self_outcome)
        self._first_node_id = f"{self._first_actor_id}_decision"
        self._second_node_id = f"{self._second_actor_id}_decision"

    def reset(self, *, rng: np.random.Generator) -> None:
        """Reset program state.

        Parameters
        ----------
        rng : numpy.random.Generator
            Runtime RNG. Unused by this stationary program.
        """

    def trial_steps(
        self,
        *,
        trial_index: int,
        trial_events: Sequence[Any],
    ) -> tuple[ProgramStep, ...]:
        """Return ordered steps for the configured timing variant."""

        del trial_index, trial_events
        steps = [
            ProgramStep(EventPhase.OBSERVATION, node_id=self._first_node_id, actor_id=self._first_actor_id),
            ProgramStep(EventPhase.DECISION, node_id=self._first_node_id, actor_id=self._first_actor_id),
        ]
        if self._first_outcome_observed:
            steps.append(
                ProgramStep(EventPhase.OUTCOME, node_id=self._first_node_id, actor_id=self._first_actor_id)
            )
            steps.append(
                ProgramStep(
                    EventPhase.UPDATE,
                    node_id=self._first_node_id,
                    actor_id=self._first_actor_id,
                    learner_id=self._first_actor_id,
                )
            )
        steps.append(
            ProgramStep(
                EventPhase.UPDATE,
                node_id=self._first_node_id,
                actor_id=self._first_actor_id,
                learner_id=self._second_actor_id,
            )
        )
        steps.extend(
            (
                ProgramStep(EventPhase.OBSERVATION, node_id=self._second_node_id, actor_id=self._second_actor_id),
                ProgramStep(EventPhase.DECISION, node_id=self._second_node_id, actor_id=self._second_actor_id),
            )
        )
        if self._second_self_outcome:
            steps.extend(
                (
                    ProgramStep(EventPhase.OUTCOME, node_id=self._second_node_id, actor_id=self._second_actor_id),
                    ProgramStep(
                        EventPhase.UPDATE,
                        node_id=self._second_node_id,
                        actor_id=self._second_actor_id,
                        learner_id=self._second_actor_id,
                    ),
                )
            )
        return tuple(steps)

    def available_actions(
        self,
        *,
        trial_index: int,
        step: ProgramStep,
        trial_events: Sequence[Any],
    ) -> tuple[int, ...]:
        """Return legal actions for the step/trial."""

        del step, trial_events
        if self._action_schedule is None:
            return self._all_actions
        return self._action_schedule[trial_index]

    def observe(
        self,
        *,
        trial_index: int,
        step: ProgramStep,
        context: DecisionContext[Any],
        trial_events: Sequence[Any],
    ) -> dict[str, Any]:
        """Return step-specific observation payload."""

        del context
        if step.actor_id == self._first_actor_id:
            return {"trial_index": trial_index, "stage": self._first_actor_id}

        first_action = _find_payload_value(
            trial_events=trial_events,
            phase_name="decision",
            key="action",
            node_id=self._first_node_id,
        )
        observation = {
            "trial_index": trial_index,
            "stage": self._second_actor_id,
            f"{self._first_actor_id}_action": first_action,
        }
        if self._first_outcome_observed:
            observation[f"{self._first_actor_id}_outcome"] = _find_payload_value(
                trial_events=trial_events,
                phase_name="outcome",
                key="outcome",
                node_id=self._first_node_id,
            )
        return observation

    def transition(
        self,
        action: Any,
        *,
        trial_index: int,
        step: ProgramStep,
        context: DecisionContext[Any],
        trial_events: Sequence[Any],
        rng: np.random.Generator,
    ) -> TwoActorSocialBanditOutcome:
        """Apply selected action and return Bernoulli outcome."""

        del trial_index, trial_events
        if action not in context.available_actions:
            raise ValueError(f"action {action!r} is not available for node {step.node_id!r}")

        p_reward = self._reward_probabilities[int(action)]
        reward = float(rng.random() < p_reward)
        return TwoActorSocialBanditOutcome(
            reward=reward,
            reward_probability=p_reward,
            source_actor_id=step.actor_id,
        )


class DemonstratorThenSubjectActionOnlyProgram(_TwoActorOrderedSocialBanditProgram):
    """Demonstrator action observed by subject before subject choice."""

    def __init__(
        self,
        reward_probabilities: Sequence[float],
        action_schedule: Sequence[Sequence[int]] | None = None,
    ) -> None:
        super().__init__(
            reward_probabilities,
            first_actor_id="demonstrator",
            second_actor_id="subject",
            first_outcome_observed=False,
            second_self_outcome=False,
            action_schedule=action_schedule,
        )


class DemonstratorThenSubjectActionOnlySelfOutcomeProgram(_TwoActorOrderedSocialBanditProgram):
    """Demonstrator action observed by subject plus subject private outcome."""

    def __init__(
        self,
        reward_probabilities: Sequence[float],
        action_schedule: Sequence[Sequence[int]] | None = None,
    ) -> None:
        super().__init__(
            reward_probabilities,
            first_actor_id="demonstrator",
            second_actor_id="subject",
            first_outcome_observed=False,
            second_self_outcome=True,
            action_schedule=action_schedule,
        )


class DemonstratorThenSubjectObservedOutcomeProgram(_TwoActorOrderedSocialBanditProgram):
    """Demonstrator action and outcome observed by subject before subject choice."""

    def __init__(
        self,
        reward_probabilities: Sequence[float],
        action_schedule: Sequence[Sequence[int]] | None = None,
    ) -> None:
        super().__init__(
            reward_probabilities,
            first_actor_id="demonstrator",
            second_actor_id="subject",
            first_outcome_observed=True,
            second_self_outcome=False,
            action_schedule=action_schedule,
        )


class DemonstratorThenSubjectObservedOutcomeSelfOutcomeProgram(_TwoActorOrderedSocialBanditProgram):
    """Demonstrator social information plus subject private-outcome learning."""

    def __init__(
        self,
        reward_probabilities: Sequence[float],
        action_schedule: Sequence[Sequence[int]] | None = None,
    ) -> None:
        super().__init__(
            reward_probabilities,
            first_actor_id="demonstrator",
            second_actor_id="subject",
            first_outcome_observed=True,
            second_self_outcome=True,
            action_schedule=action_schedule,
        )


class SubjectThenDemonstratorActionOnlyProgram(_TwoActorOrderedSocialBanditProgram):
    """Subject action observed by demonstrator before demonstrator choice."""

    def __init__(
        self,
        reward_probabilities: Sequence[float],
        action_schedule: Sequence[Sequence[int]] | None = None,
    ) -> None:
        super().__init__(
            reward_probabilities,
            first_actor_id="subject",
            second_actor_id="demonstrator",
            first_outcome_observed=False,
            second_self_outcome=False,
            action_schedule=action_schedule,
        )


class SubjectThenDemonstratorActionOnlySelfOutcomeProgram(_TwoActorOrderedSocialBanditProgram):
    """Subject action observed by demonstrator plus demonstrator private outcome."""

    def __init__(
        self,
        reward_probabilities: Sequence[float],
        action_schedule: Sequence[Sequence[int]] | None = None,
    ) -> None:
        super().__init__(
            reward_probabilities,
            first_actor_id="subject",
            second_actor_id="demonstrator",
            first_outcome_observed=False,
            second_self_outcome=True,
            action_schedule=action_schedule,
        )


class SubjectThenDemonstratorObservedOutcomeProgram(_TwoActorOrderedSocialBanditProgram):
    """Subject action and outcome observed by demonstrator before demonstration."""

    def __init__(
        self,
        reward_probabilities: Sequence[float],
        action_schedule: Sequence[Sequence[int]] | None = None,
    ) -> None:
        super().__init__(
            reward_probabilities,
            first_actor_id="subject",
            second_actor_id="demonstrator",
            first_outcome_observed=True,
            second_self_outcome=False,
            action_schedule=action_schedule,
        )


class SubjectThenDemonstratorObservedOutcomeSelfOutcomeProgram(_TwoActorOrderedSocialBanditProgram):
    """Subject social information plus demonstrator private-outcome learning."""

    def __init__(
        self,
        reward_probabilities: Sequence[float],
        action_schedule: Sequence[Sequence[int]] | None = None,
    ) -> None:
        super().__init__(
            reward_probabilities,
            first_actor_id="subject",
            second_actor_id="demonstrator",
            first_outcome_observed=True,
            second_self_outcome=True,
            action_schedule=action_schedule,
        )


def _normalize_action_schedule(
    action_schedule: Sequence[Sequence[int]] | None,
    *,
    all_actions: tuple[int, ...],
) -> tuple[tuple[int, ...], ...] | None:
    """Validate and normalize optional per-trial action subsets."""

    if action_schedule is None:
        return None

    normalized: list[tuple[int, ...]] = []
    all_action_set = set(all_actions)
    for actions in action_schedule:
        trial_actions = tuple(int(action) for action in actions)
        if not trial_actions:
            raise ValueError("each scheduled trial must expose at least one action")
        if not set(trial_actions).issubset(all_action_set):
            raise ValueError("action_schedule includes out-of-range actions")
        normalized.append(trial_actions)
    return tuple(normalized)


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


def create_demonstrator_then_subject_action_only_program(
    *,
    reward_probabilities: Sequence[float],
    action_schedule: Sequence[Sequence[int]] | None = None,
) -> DemonstratorThenSubjectActionOnlyProgram:
    """Factory used by plugin discovery."""

    return DemonstratorThenSubjectActionOnlyProgram(
        reward_probabilities=reward_probabilities,
        action_schedule=action_schedule,
    )


def create_demonstrator_then_subject_action_only_self_outcome_program(
    *,
    reward_probabilities: Sequence[float],
    action_schedule: Sequence[Sequence[int]] | None = None,
) -> DemonstratorThenSubjectActionOnlySelfOutcomeProgram:
    """Factory used by plugin discovery."""

    return DemonstratorThenSubjectActionOnlySelfOutcomeProgram(
        reward_probabilities=reward_probabilities,
        action_schedule=action_schedule,
    )


def create_demonstrator_then_subject_observed_outcome_program(
    *,
    reward_probabilities: Sequence[float],
    action_schedule: Sequence[Sequence[int]] | None = None,
) -> DemonstratorThenSubjectObservedOutcomeProgram:
    """Factory used by plugin discovery."""

    return DemonstratorThenSubjectObservedOutcomeProgram(
        reward_probabilities=reward_probabilities,
        action_schedule=action_schedule,
    )


def create_demonstrator_then_subject_observed_outcome_self_outcome_program(
    *,
    reward_probabilities: Sequence[float],
    action_schedule: Sequence[Sequence[int]] | None = None,
) -> DemonstratorThenSubjectObservedOutcomeSelfOutcomeProgram:
    """Factory used by plugin discovery."""

    return DemonstratorThenSubjectObservedOutcomeSelfOutcomeProgram(
        reward_probabilities=reward_probabilities,
        action_schedule=action_schedule,
    )


def create_subject_then_demonstrator_action_only_program(
    *,
    reward_probabilities: Sequence[float],
    action_schedule: Sequence[Sequence[int]] | None = None,
) -> SubjectThenDemonstratorActionOnlyProgram:
    """Factory used by plugin discovery."""

    return SubjectThenDemonstratorActionOnlyProgram(
        reward_probabilities=reward_probabilities,
        action_schedule=action_schedule,
    )


def create_subject_then_demonstrator_action_only_self_outcome_program(
    *,
    reward_probabilities: Sequence[float],
    action_schedule: Sequence[Sequence[int]] | None = None,
) -> SubjectThenDemonstratorActionOnlySelfOutcomeProgram:
    """Factory used by plugin discovery."""

    return SubjectThenDemonstratorActionOnlySelfOutcomeProgram(
        reward_probabilities=reward_probabilities,
        action_schedule=action_schedule,
    )


def create_subject_then_demonstrator_observed_outcome_program(
    *,
    reward_probabilities: Sequence[float],
    action_schedule: Sequence[Sequence[int]] | None = None,
) -> SubjectThenDemonstratorObservedOutcomeProgram:
    """Factory used by plugin discovery."""

    return SubjectThenDemonstratorObservedOutcomeProgram(
        reward_probabilities=reward_probabilities,
        action_schedule=action_schedule,
    )


def create_subject_then_demonstrator_observed_outcome_self_outcome_program(
    *,
    reward_probabilities: Sequence[float],
    action_schedule: Sequence[Sequence[int]] | None = None,
) -> SubjectThenDemonstratorObservedOutcomeSelfOutcomeProgram:
    """Factory used by plugin discovery."""

    return SubjectThenDemonstratorObservedOutcomeSelfOutcomeProgram(
        reward_probabilities=reward_probabilities,
        action_schedule=action_schedule,
    )


PLUGIN_MANIFESTS = [
    ComponentManifest(
        kind="problem",
        component_id="demonstrator_then_subject_action_only",
        factory=create_demonstrator_then_subject_action_only_program,
        description="Demonstrator action observed by subject before subject choice",
    ),
    ComponentManifest(
        kind="problem",
        component_id="demonstrator_then_subject_action_only_self_outcome",
        factory=create_demonstrator_then_subject_action_only_self_outcome_program,
        description="Demonstrator action observed by subject plus subject private outcome",
    ),
    ComponentManifest(
        kind="problem",
        component_id="demonstrator_then_subject_observed_outcome",
        factory=create_demonstrator_then_subject_observed_outcome_program,
        description="Demonstrator action and outcome observed by subject before subject choice",
    ),
    ComponentManifest(
        kind="problem",
        component_id="demonstrator_then_subject_observed_outcome_self_outcome",
        factory=create_demonstrator_then_subject_observed_outcome_self_outcome_program,
        description="Demonstrator social information plus subject private outcome",
    ),
    ComponentManifest(
        kind="problem",
        component_id="subject_then_demonstrator_action_only",
        factory=create_subject_then_demonstrator_action_only_program,
        description="Subject action observed by demonstrator before demonstrator choice",
    ),
    ComponentManifest(
        kind="problem",
        component_id="subject_then_demonstrator_action_only_self_outcome",
        factory=create_subject_then_demonstrator_action_only_self_outcome_program,
        description="Subject action observed by demonstrator plus demonstrator private outcome",
    ),
    ComponentManifest(
        kind="problem",
        component_id="subject_then_demonstrator_observed_outcome",
        factory=create_subject_then_demonstrator_observed_outcome_program,
        description="Subject action and outcome observed by demonstrator before demonstration",
    ),
    ComponentManifest(
        kind="problem",
        component_id="subject_then_demonstrator_observed_outcome_self_outcome",
        factory=create_subject_then_demonstrator_observed_outcome_self_outcome_program,
        description="Subject social information plus demonstrator private outcome",
    ),
]


__all__ = [
    "DemonstratorThenSubjectActionOnlyProgram",
    "DemonstratorThenSubjectActionOnlySelfOutcomeProgram",
    "DemonstratorThenSubjectObservedOutcomeProgram",
    "DemonstratorThenSubjectObservedOutcomeSelfOutcomeProgram",
    "SubjectThenDemonstratorActionOnlyProgram",
    "SubjectThenDemonstratorActionOnlySelfOutcomeProgram",
    "SubjectThenDemonstratorObservedOutcomeProgram",
    "SubjectThenDemonstratorObservedOutcomeSelfOutcomeProgram",
    "TwoActorSocialBanditOutcome",
    "create_demonstrator_then_subject_action_only_program",
    "create_demonstrator_then_subject_action_only_self_outcome_program",
    "create_demonstrator_then_subject_observed_outcome_program",
    "create_demonstrator_then_subject_observed_outcome_self_outcome_program",
    "create_subject_then_demonstrator_action_only_program",
    "create_subject_then_demonstrator_action_only_self_outcome_program",
    "create_subject_then_demonstrator_observed_outcome_program",
    "create_subject_then_demonstrator_observed_outcome_self_outcome_program",
]
