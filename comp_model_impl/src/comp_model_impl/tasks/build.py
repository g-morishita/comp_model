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

from copy import deepcopy

from comp_model_core.interfaces.block_runner import BlockRunner
from comp_model_core.plans.block import BlockPlan
from comp_model_core.spec import parse_trial_specs_schedule
from comp_model_core.registry import Registry

from .block_runner_wrappers import BanditBlockRunner, SocialBanditBlockRunner


def build_runner_for_plan(
    *,
    plan: BlockPlan,
    registries: Registry
) -> BlockRunner:
    """Build a runtime block runner for one block plan."""
    copied_plan = deepcopy(plan)

    env = registries.bandits[plan.bandit_type].from_config(plan.bandit_config)

    if plan.demonstrator_type is None:
        trial_specs = parse_trial_specs_schedule(
            n_trials=int(plan.n_trials),
            raw_trial_specs=plan.trial_specs,
            is_social=False,
        )
        return BanditBlockRunner(env=env, trial_specs=trial_specs)

    if registries.demonstrators is None:
        raise ValueError("Plan requests demonstrator_type, but no DemonstratorRegistry was provided.")

    if "model" in plan.demonstrator_config:
        copied_plan.demonstrator_config["model"] = registries.models[copied_plan.demonstrator_config["model"]]

    demo = registries.demonstrators[copied_plan.demonstrator_type].from_config(bandit_cfg=copied_plan.bandit_config, demo_cfg=copied_plan.demonstrator_config)

    trial_specs = parse_trial_specs_schedule(
        n_trials=int(plan.n_trials),
        raw_trial_specs=plan.trial_specs,
        is_social=True,
    )

    return SocialBanditBlockRunner(env=env, demonstrator=demo, trial_specs=trial_specs)
