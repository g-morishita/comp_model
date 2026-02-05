"""Block-runner construction from blueprint plans.

This module provides :func:`build_runner_for_plan`, the default adapter between
serializable :class:`comp_model_core.plans.block.BlockPlan` objects and runtime
execution objects (:class:`comp_model_core.interfaces.block_runner.BlockRunner`).

In this repo version, trial interface schedules are **fully explicit**:

- Each block plan must specify a complete per-trial ``trial_specs`` schedule
  (or a template + overrides expanded at load time).
- Outcome visibility/noise is a property of the trial interface, not the environment.

Examples
--------
Asocial runner:

>>> from comp_model_core.plans.block import BlockPlan
>>> from comp_model_impl.register import make_registry
>>> plan = BlockPlan(
...     block_id="b1",
...     n_trials=2,
...     condition="c1",
...     bandit_type="BernoulliBanditEnv",
...     bandit_config={"probs": [0.2, 0.8]},
...     trial_specs=[
...         {"self_outcome": {"kind": "VERIDICAL"}, "available_actions": [0, 1]},
...         {"self_outcome": {"kind": "VERIDICAL"}, "available_actions": [0, 1]},
...     ],
... )
>>> runner = build_runner_for_plan(plan=plan, registries=make_registry())
>>> type(runner).__name__
'BanditBlockRunner'

Social runner with demonstrator:

>>> plan = BlockPlan(
...     block_id="b2",
...     n_trials=2,
...     condition="c1",
...     bandit_type="BernoulliBanditEnv",
...     bandit_config={"probs": [0.2, 0.8]},
...     trial_specs=[
...         {"self_outcome": {"kind": "VERIDICAL"}, "demo_outcome": {"kind": "VERIDICAL"}, "available_actions": [0, 1]},
...         {"self_outcome": {"kind": "VERIDICAL"}, "demo_outcome": {"kind": "VERIDICAL"}, "available_actions": [0, 1]},
...     ],
...     demonstrator_type="RLDemonstrator",
...     demonstrator_config={"model": "QRL", "params": {"alpha": 0.2, "beta": 3.0}},
... )
>>> runner = build_runner_for_plan(plan=plan, registries=make_registry())
>>> type(runner).__name__
'SocialBanditBlockRunner'
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
    """Build a runtime block runner for one block plan.

    Parameters
    ----------
    plan : BlockPlan
        Declarative block plan (from YAML/JSON or constructed in Python).
    registries : Registry
        Registry that maps bandit, demonstrator, and model names to classes.

    Returns
    -------
    BlockRunner
        A :class:`BanditBlockRunner` (asocial) or
        :class:`SocialBanditBlockRunner` (social).

    Raises
    ------
    ValueError
        If a social plan requests a demonstrator but no demonstrator registry
        is provided.

    Notes
    -----
    If ``plan.demonstrator_config`` includes ``"model": "<name>"``, the name is
    resolved through ``registries.models`` and replaced with the model class
    before instantiating the demonstrator.
    """
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
