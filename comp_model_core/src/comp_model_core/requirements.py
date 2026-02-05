"""comp_model_core.requirements

Model/task compatibility requirements.

Summary
-------
The library separates two concepts:

- **BlockPlan**: static, serializable blueprint.
- **BlockRunner**: runtime, stateful executor.

Compatibility constraints about *what information is visible when* are expressed
at the plan level via the per-trial interface schedule (:class:`~comp_model_core.spec.TrialSpec`).
To keep the core library model-agnostic, each model returns a list of small
requirement objects.

Each requirement is evaluated against:

- the plan blueprint (:class:`~comp_model_core.plans.block.BlockPlan`),
- the environment contract (:class:`~comp_model_core.spec.EnvironmentSpec`),
- the expanded per-trial schedule (:class:`~comp_model_core.spec.TrialSpec`).

Requirements must raise :class:`~comp_model_core.errors.CompatibilityError` on failure.

Notes
-----
This module intentionally contains only *declarative* compatibility checks and
does not reference model internals. Any constraints that are truly model-specific
should be encoded as requirements returned by the model itself.

See Also
--------
comp_model_core.validation : Plan-based validation entry points.
comp_model_core.spec : TrialSpec and observation-visibility contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .errors import CompatibilityError
from .plans.block import BlockPlan
from .spec import EnvironmentSpec, OutcomeObservationKind, TrialSpec


@dataclass(frozen=True, slots=True)
class Requirement:
    """Base class for model requirements.

    Parameters
    ----------
    name
        Human-readable name for the requirement.

    Notes
    -----
    Subclasses must implement :meth:`validate` and raise
    :class:`~comp_model_core.errors.CompatibilityError` when the requirement is
    not satisfied.
    """

    name: str

    def validate(
        self,
        *,
        plan: BlockPlan,
        env_spec: EnvironmentSpec,
        trial_specs: Sequence[TrialSpec],
    ) -> None:
        """Validate the requirement against a plan and its expanded schedule.

        Parameters
        ----------
        plan
            The block blueprint being validated.
        env_spec
            The environment contract for this block (e.g., social/asocial).
        trial_specs
            Expanded per-trial schedule derived from the plan.

        Raises
        ------
        CompatibilityError
            If the requirement is not satisfied.

        Notes
        -----
        The base implementation is abstract.
        """
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class PredicateRequirement(Requirement):
    """Requirement implemented by a user-supplied predicate.

    Parameters
    ----------
    name
        Human-readable name for the requirement.
    predicate
        Callable returning True if the requirement holds.
        Signature: ``(plan, env_spec, trial_specs) -> bool``.
    message
        Error message used when the predicate returns False.

    Raises
    ------
    CompatibilityError
        If the predicate returns False, or if the predicate raises an exception.

    Notes
    -----
    Any exception raised by the predicate is caught and wrapped as a
    :class:`~comp_model_core.errors.CompatibilityError` to provide consistent
    validation error reporting.
    """

    predicate: Callable[[BlockPlan, EnvironmentSpec, Sequence[TrialSpec]], bool]
    message: str

    def validate(self, *, plan: BlockPlan, env_spec: EnvironmentSpec, trial_specs: Sequence[TrialSpec]) -> None:
        """Validate via the stored predicate.

        Parameters
        ----------
        plan
            The block blueprint being validated.
        env_spec
            The environment contract for this block.
        trial_specs
            Expanded per-trial schedule derived from the plan.

        Raises
        ------
        CompatibilityError
            If the predicate returns False, or if the predicate raises.
        """
        try:
            ok = bool(self.predicate(plan, env_spec, trial_specs))
        except Exception as e:  # pragma: no cover
            raise CompatibilityError(f"{plan.block_id}: requirement '{self.name}' raised {type(e).__name__}: {e}") from e
        if not ok:
            raise CompatibilityError(f"{plan.block_id}: {self.message}")


@dataclass(frozen=True, slots=True)
class RequireSocialBlock(Requirement):
    """Require a social block (demonstrator channel present).

    Raises
    ------
    CompatibilityError
        If ``env_spec.is_social`` is False.
    """

    def __init__(self) -> None:
        super(RequireSocialBlock, self).__init__(name="RequireSocialBlock")

    def validate(self, *, plan: BlockPlan, env_spec: EnvironmentSpec, trial_specs: Sequence[TrialSpec]) -> None:
        """Validate that the block is social.

        Parameters
        ----------
        plan
            The block blueprint being validated.
        env_spec
            Environment contract; must indicate social.
        trial_specs
            Expanded per-trial schedule (unused by this requirement).

        Raises
        ------
        CompatibilityError
            If the block is not social.
        """
        if not bool(env_spec.is_social):
            raise CompatibilityError(f"{plan.block_id}: requires a social block")


@dataclass(frozen=True, slots=True)
class RequireAsocialBlock(Requirement):
    """Require an asocial block (no demonstrator channel).

    Raises
    ------
    CompatibilityError
        If ``env_spec.is_social`` is True.
    """

    def __init__(self) -> None:
        super(RequireAsocialBlock, self).__init__(name="RequireAsocialBlock")

    def validate(self, *, plan: BlockPlan, env_spec: EnvironmentSpec, trial_specs: Sequence[TrialSpec]) -> None:
        """Validate that the block is asocial.

        Parameters
        ----------
        plan
            The block blueprint being validated.
        env_spec
            Environment contract; must indicate asocial.
        trial_specs
            Expanded per-trial schedule (unused by this requirement).

        Raises
        ------
        CompatibilityError
            If the block is social.
        """
        if bool(env_spec.is_social):
            raise CompatibilityError(f"{plan.block_id}: requires an asocial block")


@dataclass(frozen=True, slots=True)
class RequireAnySelfOutcomeObservable(Requirement):
    """Require at least one trial where the subject's outcome is observable.

    Observable includes veridical or noisy observation kinds (anything but HIDDEN).

    Raises
    ------
    CompatibilityError
        If every trial has ``self_outcome.kind == HIDDEN``.
    """

    def __init__(self) -> None:
        super(RequireAnySelfOutcomeObservable, self).__init__(name="RequireAnySelfOutcomeObservable")

    def validate(self, *, plan: BlockPlan, env_spec: EnvironmentSpec, trial_specs: Sequence[TrialSpec]) -> None:
        """Validate that at least one self outcome is observable.

        Parameters
        ----------
        plan
            The block blueprint being validated.
        env_spec
            Environment contract (unused by this requirement).
        trial_specs
            Expanded per-trial schedule.

        Raises
        ------
        CompatibilityError
            If no trial makes self outcome observable.
        """
        if not any(ts.self_outcome.kind is not OutcomeObservationKind.HIDDEN for ts in trial_specs):
            raise CompatibilityError(
                f"{plan.block_id}: requires at least one trial with self outcome observable (possibly noisy)"
            )


@dataclass(frozen=True, slots=True)
class RequireAllSelfOutcomesHidden(Requirement):
    """Require the subject's outcome to be hidden on every trial.

    Raises
    ------
    CompatibilityError
        If any trial has ``self_outcome.kind != HIDDEN``.
    """

    def __init__(self) -> None:
        super(RequireAllSelfOutcomesHidden, self).__init__(name="RequireAllSelfOutcomesHidden")

    def validate(self, *, plan: BlockPlan, env_spec: EnvironmentSpec, trial_specs: Sequence[TrialSpec]) -> None:
        """Validate that all self outcomes are hidden.

        Parameters
        ----------
        plan
            The block blueprint being validated.
        env_spec
            Environment contract (unused by this requirement).
        trial_specs
            Expanded per-trial schedule.

        Raises
        ------
        CompatibilityError
            If any trial exposes self outcome (veridical or noisy).
        """
        if any(ts.self_outcome.kind is not OutcomeObservationKind.HIDDEN for ts in trial_specs):
            raise CompatibilityError(f"{plan.block_id}: requires self outcomes to be hidden on all trials")


@dataclass(frozen=True, slots=True)
class RequireAnyDemoOutcomeObservable(Requirement):
    """Require at least one trial with demonstrator outcome observable.

    Observable includes veridical or noisy observation kinds (anything but HIDDEN).

    Raises
    ------
    CompatibilityError
        If the block is not social, or if every trial has demo outcome hidden.
    """

    def __init__(self) -> None:
        super(RequireAnyDemoOutcomeObservable, self).__init__(name="RequireAnyDemoOutcomeObservable")

    def validate(self, *, plan: BlockPlan, env_spec: EnvironmentSpec, trial_specs: Sequence[TrialSpec]) -> None:
        """Validate that at least one demonstrator outcome is observable.

        Parameters
        ----------
        plan
            The block blueprint being validated.
        env_spec
            Environment contract; must indicate social.
        trial_specs
            Expanded per-trial schedule.

        Raises
        ------
        CompatibilityError
            If the block is not social, or if no trial makes demo outcome observable.
        """
        if not bool(env_spec.is_social):
            raise CompatibilityError(f"{plan.block_id}: requires a social block (demonstrator channel)")
        if not any(
            ts.demo_outcome is not None and ts.demo_outcome.kind is not OutcomeObservationKind.HIDDEN
            for ts in trial_specs
        ):
            raise CompatibilityError(
                f"{plan.block_id}: requires at least one trial with demonstrator outcome observable (possibly noisy)"
            )


@dataclass(frozen=True, slots=True)
class RequireAllDemoOutcomesHidden(Requirement):
    """Require demonstrator outcomes to be hidden on every trial.

    Raises
    ------
    CompatibilityError
        If the block is not social, or if any trial exposes demo outcome.
    """

    def __init__(self) -> None:
        super(RequireAllDemoOutcomesHidden, self).__init__(name="RequireAllDemoOutcomesHidden")

    def validate(self, *, plan: BlockPlan, env_spec: EnvironmentSpec, trial_specs: Sequence[TrialSpec]) -> None:
        """Validate that all demonstrator outcomes are hidden.

        Parameters
        ----------
        plan
            The block blueprint being validated.
        env_spec
            Environment contract; must indicate social.
        trial_specs
            Expanded per-trial schedule.

        Raises
        ------
        CompatibilityError
            If the block is not social, or if any trial exposes demo outcome.
        """
        if not bool(env_spec.is_social):
            raise CompatibilityError(f"{plan.block_id}: requires a social block (demonstrator channel)")
        if any(
            ts.demo_outcome is not None and ts.demo_outcome.kind is not OutcomeObservationKind.HIDDEN
            for ts in trial_specs
        ):
            raise CompatibilityError(f"{plan.block_id}: requires demonstrator outcomes to be hidden on all trials")
