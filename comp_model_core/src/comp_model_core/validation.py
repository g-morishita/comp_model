"""comp_model_core.validation

Compatibility validation between block runners (trial interface schedules) and models.

Validation rules fall into two buckets:

1) **Action-set validity**
   Ensures that trial-specific available action sets are non-empty and within range.

2) **Model requirements**
   Ensures that the runner provides the information required by a model at least
   once (e.g., demonstrator outcome observation for vicarious learning models).

Notes
-----
These validators are intended to fail fast during simulation or when checking plan
consistency before fitting.

See Also
--------
comp_model_core.requirements.ModelRequirements
comp_model_core.interfaces.block_runner.BlockRunner
"""

from __future__ import annotations

from .errors import CompatibilityError
from .interfaces.block_runner import BlockRunner, SocialBlockRunner
from .requirements import ModelRequirements
from .spec import OutcomeObsKind


def validate_action_sets(
    *,
    runner: BlockRunner,
    n_trials: int,
    block_id: str | None = None,
) -> None:
    """Validate per-trial action sets.

    Parameters
    ----------
    runner : BlockRunner
        Runner whose trial specs will be checked.
    n_trials : int
        Number of trials to validate (typically from the plan).
    block_id : str or None, optional
        Optional block identifier for error messages.

    Raises
    ------
    CompatibilityError
        If a trial has an empty action set or contains an out-of-range action index.
    """
    bid = f" block_id={block_id!r}" if block_id is not None else ""
    nA = int(runner.spec.n_actions)

    for t in range(int(n_trials)):
        ts = runner.resolved_trial_spec(t=t)
        if ts.available_actions is None:
            continue
        if len(ts.available_actions) == 0:
            raise CompatibilityError(f"Trial {t}{bid}: available_actions is empty")
        for a in ts.available_actions:
            ai = int(a)
            if ai < 0 or ai >= nA:
                raise CompatibilityError(f"Trial {t}{bid}: invalid action {ai}; n_actions={nA}")


def validate_runner_against_model_requirements(
    *,
    runner: BlockRunner,
    n_trials: int,
    reqs: ModelRequirements,
    block_id: str | None = None,
) -> None:
    """Validate that a runner provides what a model declares it needs.

    Parameters
    ----------
    runner : BlockRunner
        Runner to validate.
    n_trials : int
        Number of trials to scan (typically from the plan).
    reqs : ModelRequirements
        Requirements declared by the model.
    block_id : str or None, optional
        Optional block identifier for error messages.

    Raises
    ------
    CompatibilityError
        If one or more requirements are not satisfied.
    """
    bid = f" block_id={block_id!r}" if block_id is not None else ""

    resolved = [runner.resolved_trial_spec(t=t) for t in range(int(n_trials))]
    errors: list[str] = []

    if reqs.needs_self_outcome_at_least_once:
        if not any(ts.self_outcome.kind is not OutcomeObsKind.HIDDEN for ts in resolved):
            errors.append("requires at least one trial with self outcome observed (not hidden)")

    if reqs.needs_demo_choice_at_least_once or reqs.needs_demo_outcome_at_least_once:
        if not runner.spec.is_social or not isinstance(runner, SocialBlockRunner):
            errors.append("requires a social runner (demonstrator observations)")
        else:
            if reqs.needs_demo_outcome_at_least_once:
                if not any(
                    (ts.demo_outcome is not None and ts.demo_outcome.kind is not OutcomeObsKind.HIDDEN)
                    for ts in resolved
                ):
                    errors.append("requires at least one trial with demonstrator outcome observed (not hidden)")

    for name, pred in reqs.trial_predicates:
        ok = False
        try:
            ok = bool(pred(resolved))
        except Exception as e:
            errors.append(f"predicate '{name}' raised {type(e).__name__}: {e}")
        if not ok:
            errors.append(f"failed predicate '{name}'")

    if errors:
        raise CompatibilityError(
            "Runner/model compatibility errors"
            + bid
            + ":\n- "
            + "\n- ".join(errors)
        )
