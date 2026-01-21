from __future__ import annotations

from ..bandits.registry import BanditRegistry
from ..demonstrators.registry import DemonstratorRegistry
from comp_model_core.interfaces.bandit import Bandit
from comp_model_core.plans.block import BlockPlan
from .social_wrapper import SocialBanditWrapper


def build_bandit_for_plan(
    *,
    plan: BlockPlan,
    bandits: BanditRegistry,
    demonstrators: DemonstratorRegistry | None = None,
    reveal_demo_outcome: bool = False,
) -> Bandit:
    """
    Build a (possibly social) bandit for one block plan.
    """
    base = bandits.make(plan.bandit_type, plan.bandit_config)

    if plan.demonstrator_type is None:
        return base

    if demonstrators is None:
        raise ValueError("Plan requests demonstrator_type, but no DemonstratorRegistry was provided.")

    demo = demonstrators.make(
        plan.demonstrator_type,
        bandit_cfg=plan.bandit_config,
        demo_cfg=plan.demonstrator_config or {},
    )

    return SocialBanditWrapper(base=base, demonstrator=demo, reveal_demo_outcome=reveal_demo_outcome)
