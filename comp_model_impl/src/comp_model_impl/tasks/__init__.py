"""Task and bandit construction utilities.

This package provides helpers to build executable block runners from declarative
plans, wiring together bandit environments, trial specifications, and optional
demonstrators.

Examples
--------
Build a runner from a :class:`~comp_model_core.plans.block.BlockPlan`:

>>> from comp_model_core.plans.block import BlockPlan
>>> from comp_model_impl.register import make_registry
>>> from comp_model_impl.tasks import build_runner_for_plan
>>> plan = BlockPlan(
...     block_id="b1",
...     n_trials=3,
...     condition="c1",
...     bandit_type="BernoulliBanditEnv",
...     bandit_config={"probs": [0.2, 0.8]},
...     trial_specs=[{"self_outcome": {"kind": "VERIDICAL"}, "available_actions": [0, 1]}] * 3,
... )
>>> runner = build_runner_for_plan(plan, registries=make_registry())
>>> type(runner).__name__ in {"BanditBlockRunner", "SocialBanditBlockRunner"}
True

See Also
--------
comp_model_impl.tasks.build.build_runner_for_plan
comp_model_impl.tasks.block_runner_wrappers
"""

from .build import build_runner_for_plan
from .block_runner_wrappers import BanditBlockRunner, SocialBanditBlockRunner

__all__ = [
    "build_runner_for_plan",
    "BanditBlockRunner",
    "SocialBanditBlockRunner",
]
