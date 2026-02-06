"""comp_model_core.validation

Plan-level compatibility validation.

Validation in this library is intentionally **plan-based**:

- Trial interface schedules live in :class:`comp_model_core.plans.block.BlockPlan`.
- Runners are runtime executors; validation should not require runner state.

The core library provides model-agnostic checks (e.g., action-set validity) and
delegates model-specific checks to small :class:`comp_model_core.requirements.Requirement`
objects returned by each model.
"""

from __future__ import annotations

from typing import Sequence

from .errors import CompatibilityError
from .plans.block import BlockPlan
from .requirements import Requirement
from .spec import EnvironmentSpec, TrialSpec, parse_trial_specs_schedule


def validate_action_sets(
    *,
    trial_specs: Sequence[TrialSpec],
    env_spec: EnvironmentSpec,
    block_id: str | None = None,
) -> None:
    """Validate that per-trial action sets (if present) are non-empty and in-range."""

    bid = f" block_id={block_id!r}" if block_id is not None else ""
    nA = int(env_spec.n_actions)

    for t, ts in enumerate(trial_specs):
        if ts.available_actions is None:
            continue
        if len(ts.available_actions) == 0:
            raise CompatibilityError(f"Trial {t}{bid}: available_actions is empty")
        for a in ts.available_actions:
            ai = int(a)
            if ai < 0 or ai >= nA:
                raise CompatibilityError(f"Trial {t}{bid}: invalid action {ai}; n_actions={nA}")


def validate_action_coverage(
    *,
    trial_specs: Sequence[TrialSpec],
    env_spec: EnvironmentSpec,
    block_id: str | None = None,
) -> None:
    """Validate that all actions appear at least once if available_actions is specified."""
    bid = f" block_id={block_id!r}" if block_id is not None else ""
    nA = int(env_spec.n_actions)

    seen: set[int] = set()
    saw_mask = False
    for ts in trial_specs:
        if ts.available_actions is None:
            continue
        saw_mask = True
        for a in ts.available_actions:
            seen.add(int(a))

    if saw_mask and len(seen) < nA:
        missing = sorted(set(range(nA)) - seen)
        raise CompatibilityError(
            f"{bid}: available_actions excludes actions for the entire block. "
            f"Missing actions: {missing}. "
            "Either include all actions in available_actions or adjust n_actions/bandit_config."
        )


def validate_block_plan(
    *,
    plan: BlockPlan,
    env_spec: EnvironmentSpec,
    requirements: Sequence[Requirement] = (),
) -> list[TrialSpec]:
    """Validate a plan against the environment contract and model requirements.

    Parameters
    ----------
    plan:
        Block blueprint.
    env_spec:
        Environment contract for the instantiated runner/environment.
    requirements:
        Model requirements to evaluate for this plan.

    Returns
    -------
    list[TrialSpec]
        Parsed trial schedule for convenience.
    """

    trial_specs = parse_trial_specs_schedule(
        n_trials=int(plan.n_trials),
        raw_trial_specs=plan.trial_specs,
        is_social=bool(env_spec.is_social),
    )

    validate_action_sets(trial_specs=trial_specs, env_spec=env_spec, block_id=plan.block_id)
    validate_action_coverage(trial_specs=trial_specs, env_spec=env_spec, block_id=plan.block_id)

    for req in requirements:
        req.validate(plan=plan, env_spec=env_spec, trial_specs=trial_specs)

    return trial_specs
