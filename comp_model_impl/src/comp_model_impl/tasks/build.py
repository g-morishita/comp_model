"""comp_model_impl.tasks.build

Block-runner construction from blueprint plans.

This module provides :func:`build_runner_for_plan`, the default adapter between
serializable :class:`~comp_model_core.plans.block.BlockPlan` objects and runtime
execution objects (:class:`~comp_model_core.interfaces.block_runner.BlockRunner`).

Defaults
--------
The builder accepts block-level defaults for outcome observation models. For convenience:

- If you pass a boolean:
  - ``True``  -> veridical observation (kind=VERIDICAL)
  - ``False`` -> hidden outcome (kind=HIDDEN)

- You may also pass YAML/JSON-style dicts such as:
  ``{"kind": "GAUSSIAN", "sigma": 0.1}``

See Also
--------
comp_model_core.plans.block.BlockPlan
comp_model_core.interfaces.block_runner.BlockRunner
comp_model_core.spec.OutcomeObservationSpec
comp_model_impl.tasks.social_wrapper.BanditBlockRunner
comp_model_impl.tasks.social_wrapper.SocialBanditBlockRunner
"""

from __future__ import annotations

from comp_model_core.interfaces.block_runner import BlockRunner
from comp_model_core.plans.block import BlockPlan
from comp_model_core.spec import (
    OutcomeObservationKind,
    OutcomeObservationSpec,
    OutcomeObservationLike,
    parse_outcome_observation,
)

from ..bandits.registry import BanditRegistry
from ..demonstrators.registry import DemonstratorRegistry
from .runner_block_wrappers import BanditBlockRunner, SocialBanditBlockRunner, parse_trial_specs


def _coerce_default_obs(x: OutcomeObservationLike | bool | None) -> OutcomeObservationSpec:
    """Coerce a default outcome observation specification.

    Parameters
    ----------
    x : OutcomeObservationLike or bool or None
        Default observation specification.
        - None -> VERIDICAL
        - bool -> VERIDICAL if True else HIDDEN
        - mapping/spec -> parsed via :func:`comp_model_core.spec.parse_outcome_observation`

    Returns
    -------
    OutcomeObservationSpec
        Concrete observation specification.

    Raises
    ------
    TypeError
        If ``x`` is not a supported type.
    ValueError
        If parsing fails (e.g., invalid kind or missing parameters).
    """
    if x is None:
        return OutcomeObservationSpec(kind=OutcomeObservationKind.VERIDICAL)
    if isinstance(x, bool):
        return OutcomeObservationSpec(
            kind=OutcomeObservationKind.VERIDICAL if x else OutcomeObservationKind.HIDDEN
        )
    return parse_outcome_observation(x)


def build_runner_for_plan(
    *,
    plan: BlockPlan,
    bandits: BanditRegistry,
    demonstrators: DemonstratorRegistry | None = None,
    default_self_outcome: OutcomeObservationLike | bool | None = True,
    default_demo_outcome: OutcomeObservationLike | bool | None = True,
) -> BlockRunner:
    """Build a runtime block runner for one block plan.

    Parameters
    ----------
    plan : BlockPlan
        Blueprint for a block.
    bandits : BanditRegistry
        Registry for environment (bandit) implementations.
    demonstrators : DemonstratorRegistry or None, optional
        Registry for demonstrator implementations. Required when the plan requests
        a demonstrator.
    default_self_outcome : OutcomeObservationLike or bool or None, optional
        Block-level default for how the subject observes its own outcome.
        If ``None`` or ``True``, defaults to veridical observation.
        If ``False``, defaults to hidden outcomes.
    default_demo_outcome : OutcomeObservationLike or bool or None, optional
        Block-level default for how the subject observes demonstrator outcomes.
        Interpreted the same way as ``default_self_outcome``. Only used for social blocks.

    Returns
    -------
    BlockRunner
        Runtime runner for the block. Returns:
        - :class:`~comp_model_impl.tasks.social_wrapper.BanditBlockRunner` for asocial blocks
        - :class:`~comp_model_impl.tasks.social_wrapper.SocialBanditBlockRunner` for social blocks

    Raises
    ------
    ValueError
        If the plan requests a demonstrator but no demonstrator registry is provided.
    """
    env = bandits.make(plan.bandit_type, plan.bandit_config)
    self_default = _coerce_default_obs(default_self_outcome)

    # Asocial block
    if plan.demonstrator_type is None:
        resolved = parse_trial_specs(
            n_trials=int(plan.n_trials),
            raw_trial_specs=plan.trial_specs,
            default_self_outcome=self_default,
            default_demo_outcome=None,
        )
        return BanditBlockRunner(env=env, resolved_trial_specs=resolved)

    # Social block
    if demonstrators is None:
        raise ValueError("Plan requests demonstrator_type, but no DemonstratorRegistry was provided.")

    demo = demonstrators.make(
        plan.demonstrator_type,
        bandit_cfg=plan.bandit_config,
        demo_cfg=plan.demonstrator_config or {},
    )

    demo_default = _coerce_default_obs(default_demo_outcome)
    resolved = parse_trial_specs(
        n_trials=int(plan.n_trials),
        raw_trial_specs=plan.trial_specs,
        default_self_outcome=self_default,
        default_demo_outcome=demo_default,
    )

    return SocialBanditBlockRunner(env=env, demonstrator=demo, resolved_trial_specs=resolved)
