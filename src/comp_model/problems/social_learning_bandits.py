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
    """Internal base for ordered two-actor social bandit variants.

    This class parameterizes the eight public social-learning timing presets in
    this module. The actor order can flip, but the role semantics do not:

    - ``subject`` is always the learner of interest,
    - ``demonstrator`` is always the source of social information,
    - the subject may act either before or after the demonstrator.

    Each trial contains one subject decision node and one demonstrator decision
    node. The subject always receives the social update from the demonstrator
    node, regardless of whether the subject acts first or second in the trial.
    When the demonstrator acts first, that social update can influence the same
    trial's subject choice. When the subject acts first, the social update
    occurs after the subject's own choice and therefore affects later behavior.

    ``reward_probabilities`` gives one Bernoulli reward rate per action. These
    are reward rates for the environment, not a probability distribution over
    actions, so repeated values such as ``[0.5, 0.5]`` are valid.

    This is an internal configurable base. External code should use one of the
    named public program classes instead of instantiating this class directly.
    """

    def __init__(
        self,
        reward_probabilities: Sequence[float],
        *,
        subject_first: bool,
        demonstrator_outcome_observed: bool,
        subject_self_outcome: bool,
        action_schedule: Sequence[Sequence[int]] | None = None,
    ) -> None:
        if len(reward_probabilities) == 0:
            raise ValueError("reward_probabilities must contain at least one action")

        self._reward_probabilities = tuple(float(value) for value in reward_probabilities)
        for value in self._reward_probabilities:
            if value < 0.0 or value > 1.0:
                raise ValueError("reward probabilities must be within [0, 1]")

        self._all_actions = tuple(range(len(self._reward_probabilities)))
        self._action_schedule = _normalize_action_schedule(action_schedule, all_actions=self._all_actions)
        self._subject_first = bool(subject_first)
        self._demonstrator_outcome_observed = bool(demonstrator_outcome_observed)
        self._subject_self_outcome = bool(subject_self_outcome)
        self._subject_actor_id = "subject"
        self._demonstrator_actor_id = "demonstrator"
        self._subject_decision_node_id = "subject_decision"
        self._demonstrator_decision_node_id = "demonstrator_decision"

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
        if self._subject_first:
            steps = list(self._subject_node_steps(include_self_outcome=self._subject_self_outcome))
            steps.extend(self._demonstrator_node_steps(include_outcome=self._demonstrator_outcome_observed))
        else:
            steps = list(self._demonstrator_node_steps(include_outcome=self._demonstrator_outcome_observed))
            steps.extend(self._subject_node_steps(include_self_outcome=self._subject_self_outcome))
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
        if step.actor_id == self._subject_actor_id:
            if self._subject_first:
                return {"trial_index": trial_index, "stage": self._subject_actor_id}
            return self._subject_observation_after_demonstrator(
                trial_index=trial_index,
                trial_events=trial_events,
            )

        if self._subject_first:
            return self._demonstrator_observation_after_subject(
                trial_index=trial_index,
                trial_events=trial_events,
            )
        return {"trial_index": trial_index, "stage": self._demonstrator_actor_id}

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
            raise ValueError(
                f"action {action!r} is not available for decision node "
                f"{step.decision_node_id!r}"
            )

        p_reward = self._reward_probabilities[int(action)]
        reward = float(rng.random() < p_reward)
        return TwoActorSocialBanditOutcome(
            reward=reward,
            reward_probability=p_reward,
            source_actor_id=step.actor_id,
        )

    def _subject_node_steps(self, *, include_self_outcome: bool) -> tuple[ProgramStep, ...]:
        """Return subject-node steps in fixed role semantics."""

        steps: list[ProgramStep] = [
            ProgramStep(
                EventPhase.OBSERVATION,
                decision_node_id=self._subject_decision_node_id,
                actor_id=self._subject_actor_id,
            ),
            ProgramStep(
                EventPhase.DECISION,
                decision_node_id=self._subject_decision_node_id,
                actor_id=self._subject_actor_id,
            ),
        ]
        if include_self_outcome:
            steps.extend(
                (
                    ProgramStep(
                        EventPhase.OUTCOME,
                        decision_node_id=self._subject_decision_node_id,
                        actor_id=self._subject_actor_id,
                    ),
                    ProgramStep(
                        EventPhase.UPDATE,
                        decision_node_id=self._subject_decision_node_id,
                        actor_id=self._subject_actor_id,
                        learner_id=self._subject_actor_id,
                    ),
                )
            )
        return tuple(steps)

    def _demonstrator_node_steps(self, *, include_outcome: bool) -> tuple[ProgramStep, ...]:
        """Return demonstrator-node steps plus the subject's social update."""

        steps: list[ProgramStep] = [
            ProgramStep(
                EventPhase.OBSERVATION,
                decision_node_id=self._demonstrator_decision_node_id,
                actor_id=self._demonstrator_actor_id,
            ),
            ProgramStep(
                EventPhase.DECISION,
                decision_node_id=self._demonstrator_decision_node_id,
                actor_id=self._demonstrator_actor_id,
            ),
        ]
        if include_outcome:
            steps.extend(
                (
                    ProgramStep(
                        EventPhase.OUTCOME,
                        decision_node_id=self._demonstrator_decision_node_id,
                        actor_id=self._demonstrator_actor_id,
                    ),
                    ProgramStep(
                        EventPhase.UPDATE,
                        decision_node_id=self._demonstrator_decision_node_id,
                        actor_id=self._demonstrator_actor_id,
                        learner_id=self._demonstrator_actor_id,
                    ),
                )
            )
        # Subject is always the downstream learner for demonstrator behavior in
        # this social-task family, independent of whether the subject acted
        # earlier or later in the same trial.
        steps.append(
            ProgramStep(
                EventPhase.UPDATE,
                decision_node_id=self._demonstrator_decision_node_id,
                actor_id=self._demonstrator_actor_id,
                learner_id=self._subject_actor_id,
            )
        )
        return tuple(steps)

    def _subject_observation_after_demonstrator(
        self,
        *,
        trial_index: int,
        trial_events: Sequence[Any],
    ) -> dict[str, Any]:
        """Build subject observation when demonstrator acts first."""

        demonstrator_action = _find_payload_value(
            trial_events=trial_events,
            phase_name="decision",
            key="action",
            decision_node_id=self._demonstrator_decision_node_id,
        )
        observation = {
            "trial_index": trial_index,
            "stage": self._subject_actor_id,
            "demonstrator_action": demonstrator_action,
        }
        if self._demonstrator_outcome_observed:
            observation["demonstrator_outcome"] = _find_payload_value(
                trial_events=trial_events,
                phase_name="outcome",
                key="outcome",
                decision_node_id=self._demonstrator_decision_node_id,
            )
        return observation

    def _demonstrator_observation_after_subject(
        self,
        *,
        trial_index: int,
        trial_events: Sequence[Any],
    ) -> dict[str, Any]:
        """Build demonstrator observation when subject acts first."""

        subject_action = _find_payload_value(
            trial_events=trial_events,
            phase_name="decision",
            key="action",
            decision_node_id=self._subject_decision_node_id,
        )
        observation = {
            "trial_index": trial_index,
            "stage": self._demonstrator_actor_id,
            "subject_action": subject_action,
        }
        if self._subject_self_outcome:
            observation["subject_outcome"] = _find_payload_value(
                trial_events=trial_events,
                phase_name="outcome",
                key="outcome",
                decision_node_id=self._subject_decision_node_id,
            )
        return observation


class DemonstratorThenSubjectActionOnlyProgram(_TwoActorOrderedSocialBanditProgram):
    """Demonstrator action observed by subject before subject choice."""

    def __init__(
        self,
        reward_probabilities: Sequence[float],
        action_schedule: Sequence[Sequence[int]] | None = None,
    ) -> None:
        super().__init__(
            reward_probabilities,
            subject_first=False,
            demonstrator_outcome_observed=False,
            subject_self_outcome=False,
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
            subject_first=False,
            demonstrator_outcome_observed=False,
            subject_self_outcome=True,
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
            subject_first=False,
            demonstrator_outcome_observed=True,
            subject_self_outcome=False,
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
            subject_first=False,
            demonstrator_outcome_observed=True,
            subject_self_outcome=True,
            action_schedule=action_schedule,
        )


class SubjectThenDemonstratorActionOnlyProgram(_TwoActorOrderedSocialBanditProgram):
    """Subject acts first; later demonstrator action updates the subject only."""

    def __init__(
        self,
        reward_probabilities: Sequence[float],
        action_schedule: Sequence[Sequence[int]] | None = None,
    ) -> None:
        super().__init__(
            reward_probabilities,
            subject_first=True,
            demonstrator_outcome_observed=False,
            subject_self_outcome=False,
            action_schedule=action_schedule,
        )


class SubjectThenDemonstratorActionOnlySelfOutcomeProgram(_TwoActorOrderedSocialBanditProgram):
    """Subject acts first with private outcome, then learns from demonstrator action."""

    def __init__(
        self,
        reward_probabilities: Sequence[float],
        action_schedule: Sequence[Sequence[int]] | None = None,
    ) -> None:
        super().__init__(
            reward_probabilities,
            subject_first=True,
            demonstrator_outcome_observed=False,
            subject_self_outcome=True,
            action_schedule=action_schedule,
        )


class SubjectThenDemonstratorObservedOutcomeProgram(_TwoActorOrderedSocialBanditProgram):
    """Subject acts first, then later learns from demonstrator action and outcome."""

    def __init__(
        self,
        reward_probabilities: Sequence[float],
        action_schedule: Sequence[Sequence[int]] | None = None,
    ) -> None:
        super().__init__(
            reward_probabilities,
            subject_first=True,
            demonstrator_outcome_observed=True,
            subject_self_outcome=False,
            action_schedule=action_schedule,
        )


class SubjectThenDemonstratorObservedOutcomeSelfOutcomeProgram(_TwoActorOrderedSocialBanditProgram):
    """Subject gets private outcome, then later learns from demonstrator outcome."""

    def __init__(
        self,
        reward_probabilities: Sequence[float],
        action_schedule: Sequence[Sequence[int]] | None = None,
    ) -> None:
        super().__init__(
            reward_probabilities,
            subject_first=True,
            demonstrator_outcome_observed=True,
            subject_self_outcome=True,
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
    decision_node_id: str,
) -> Any:
    """Find payload value for a specific phase/node in trial events."""

    for event in trial_events:
        if getattr(event.phase, "value", None) != phase_name:
            continue
        payload = getattr(event, "payload", None)
        if not isinstance(payload, dict):
            continue
        if payload.get("decision_node_id") != decision_node_id:
            continue
        if key in payload:
            return payload[key]

    raise ValueError(
        "trial_events missing payload key "
        f"{key!r} for phase {phase_name!r} and decision node {decision_node_id!r}"
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
