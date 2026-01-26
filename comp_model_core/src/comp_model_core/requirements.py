"""comp_model_core.requirements

Model/task compatibility requirements.

The library separates:

- **BlockPlan**: static, serializable blueprint.
- **BlockRunner**: runtime, stateful executor.

Compatibility constraints about *what information is visible when* are properties
of the plan's per-trial interface schedule (``TrialSpec``). To keep the core
library model-agnostic, each model returns a list of small requirement objects.

A requirement is evaluated against:

- the :class:`~comp_model_core.plans.block.BlockPlan` (blueprint),
- the environment contract (:class:`~comp_model_core.spec.EnvironmentSpec`),
- the expanded per-trial schedule (:class:`~comp_model_core.spec.TrialSpec`).

Requirements should raise :class:`~comp_model_core.errors.CompatibilityError` on
failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .errors import CompatibilityError
from .plans.block import BlockPlan
from .spec import EnvironmentSpec, OutcomeObservationKind, TrialSpec


@dataclass(frozen=True, slots=True)
class Requirement:
    """Base class for model requirements."""

    name: str

    def validate(
        self,
        *,
        plan: BlockPlan,
        env_spec: EnvironmentSpec,
        trial_specs: Sequence[TrialSpec],
    ) -> None:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class PredicateRequirement(Requirement):
    """Requirement implemented by a user-supplied predicate."""

    predicate: Callable[[BlockPlan, EnvironmentSpec, Sequence[TrialSpec]], bool]
    message: str

    def validate(self, *, plan: BlockPlan, env_spec: EnvironmentSpec, trial_specs: Sequence[TrialSpec]) -> None:
        try:
            ok = bool(self.predicate(plan, env_spec, trial_specs))
        except Exception as e:  # pragma: no cover
            raise CompatibilityError(f"{plan.block_id}: requirement '{self.name}' raised {type(e).__name__}: {e}") from e
        if not ok:
            raise CompatibilityError(f"{plan.block_id}: {self.message}")


@dataclass(frozen=True, slots=True)
class RequireSocialBlock(Requirement):
    """Require a social block (demonstrator channel)."""

    def __init__(self) -> None:
        super(RequireSocialBlock, self).__init__(name="RequireSocialBlock")

    def validate(self, *, plan: BlockPlan, env_spec: EnvironmentSpec, trial_specs: Sequence[TrialSpec]) -> None:
        if not bool(env_spec.is_social):
            raise CompatibilityError(f"{plan.block_id}: requires a social block")


@dataclass(frozen=True, slots=True)
class RequireAsocialBlock(Requirement):
    """Require an asocial block."""

    def __init__(self) -> None:
        super(RequireAsocialBlock, self).__init__(name="RequireAsocialBlock")

    def validate(self, *, plan: BlockPlan, env_spec: EnvironmentSpec, trial_specs: Sequence[TrialSpec]) -> None:
        if bool(env_spec.is_social):
            raise CompatibilityError(f"{plan.block_id}: requires an asocial block")


@dataclass(frozen=True, slots=True)
class RequireAnySelfOutcomeObservable(Requirement):
    """Require at least one trial where the subject's outcome is observable."""

    def __init__(self) -> None:
        super(RequireAnySelfOutcomeObservable, self).__init__(name="RequireAnySelfOutcomeObservable")

    def validate(self, *, plan: BlockPlan, env_spec: EnvironmentSpec, trial_specs: Sequence[TrialSpec]) -> None:
        if not any(ts.self_outcome.kind is not OutcomeObservationKind.HIDDEN for ts in trial_specs):
            raise CompatibilityError(
                f"{plan.block_id}: requires at least one trial with self outcome observable (possibly noisy)"
            )


@dataclass(frozen=True, slots=True)
class RequireAllSelfOutcomesHidden(Requirement):
    """Require the subject's outcome to be hidden on every trial."""

    def __init__(self) -> None:
        super(RequireAllSelfOutcomesHidden, self).__init__(name="RequireAllSelfOutcomesHidden")

    def validate(self, *, plan: BlockPlan, env_spec: EnvironmentSpec, trial_specs: Sequence[TrialSpec]) -> None:
        if any(ts.self_outcome.kind is not OutcomeObservationKind.HIDDEN for ts in trial_specs):
            raise CompatibilityError(f"{plan.block_id}: requires self outcomes to be hidden on all trials")


@dataclass(frozen=True, slots=True)
class RequireAnyDemoOutcomeObservable(Requirement):
    """Require at least one trial with demonstrator outcome observable."""

    def __init__(self) -> None:
        super(RequireAnyDemoOutcomeObservable, self).__init__(name="RequireAnyDemoOutcomeObservable")

    def validate(self, *, plan: BlockPlan, env_spec: EnvironmentSpec, trial_specs: Sequence[TrialSpec]) -> None:
        if not bool(env_spec.is_social):
            raise CompatibilityError(f"{plan.block_id}: requires a social block (demonstrator channel)")
        if not any(ts.demo_outcome is not None and ts.demo_outcome.kind is not OutcomeObservationKind.HIDDEN for ts in trial_specs):
            raise CompatibilityError(
                f"{plan.block_id}: requires at least one trial with demonstrator outcome observable (possibly noisy)"
            )


@dataclass(frozen=True, slots=True)
class RequireAllDemoOutcomesHidden(Requirement):
    """Require demonstrator outcomes to be hidden on every trial."""

    def __init__(self) -> None:
        super(RequireAllDemoOutcomesHidden, self).__init__(name="RequireAllDemoOutcomesHidden")

    def validate(self, *, plan: BlockPlan, env_spec: EnvironmentSpec, trial_specs: Sequence[TrialSpec]) -> None:
        if not bool(env_spec.is_social):
            raise CompatibilityError(f"{plan.block_id}: requires a social block (demonstrator channel)")
        if any(ts.demo_outcome is not None and ts.demo_outcome.kind is not OutcomeObservationKind.HIDDEN for ts in trial_specs):
            raise CompatibilityError(f"{plan.block_id}: requires demonstrator outcomes to be hidden on all trials")
