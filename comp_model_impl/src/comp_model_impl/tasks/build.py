"""comp_model_impl.tasks.build

Block-runner construction from blueprint plans.

This module provides :func:`build_runner_for_plan`, the default adapter between
serializable :class:`comp_model_core.plans.block.BlockPlan` objects and runtime
execution objects (:class:`comp_model_core.interfaces.block_runner.BlockRunner`).

In this repo version, trial interface schedules are **fully explicit**:

- Each block plan must specify a complete per-trial ``trial_specs`` schedule
  (or a template + overrides expanded at load time).
- Outcome visibility/noise is a property of the trial interface, not the environment.
"""

from __future__ import annotations

from comp_model_core.interfaces.block_runner import BlockRunner
from comp_model_core.plans.block import BlockPlan
from comp_model_core.spec import parse_trial_specs_schedule
from comp_model_core.registry import NamedRegistry
from comp_model_core.interfaces.bandit import BanditEnv

from ..demonstrators.registry import DemonstratorRegistry
from .block_runner_wrappers import BanditBlockRunner, SocialBanditBlockRunner


def build_runner_for_plan(
    *,
    plan: BlockPlan,
    bandits: NamedRegistry[BanditEnv],
    demonstrators: DemonstratorRegistry | None = None,
) -> BlockRunner:
    """Build a runtime block runner for one block plan."""

    env = bandits[plan.bandit_type].from_config(plan.bandit_config)

    if plan.demonstrator_type is None:
        trial_specs = parse_trial_specs_schedule(
            n_trials=int(plan.n_trials),
            raw_trial_specs=plan.trial_specs,
            is_social=False,
        )
        return BanditBlockRunner(env=env, trial_specs=trial_specs)

    if demonstrators is None:
        raise ValueError("Plan requests demonstrator_type, but no DemonstratorRegistry was provided.")

    demo = demonstrators.make(
        plan.demonstrator_type,
        bandit_cfg=plan.bandit_config,
        demo_cfg=plan.demonstrator_config or {},
    )

    trial_specs = parse_trial_specs_schedule(
        n_trials=int(plan.n_trials),
        raw_trial_specs=plan.trial_specs,
        is_social=True,
    )

    return SocialBanditBlockRunner(env=env, demonstrator=demo, trial_specs=trial_specs)
