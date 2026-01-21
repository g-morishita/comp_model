from __future__ import annotations

from typing import Any, Mapping

from .noisy_best import NoisyBestArmDemonstrator
from .fixed_sequence import FixedSequenceDemonstrator
from .rl_agent import RLDemonstrator
from comp_model_core.interfaces.demonstrator import Demonstrator
from comp_model_core.interfaces.model import ComputationalModel


def make_noisy_best(bandit_cfg: Mapping[str, Any], demo_cfg: Mapping[str, Any]) -> Demonstrator:
    probs = bandit_cfg["probs"]  # required for Bernoulli bandit config
    p_best = float(demo_cfg.get("p_best", 0.8))
    return NoisyBestArmDemonstrator(reward_probs=probs, p_best=p_best)


def make_fixed_sequence(bandit_cfg: Mapping[str, Any], demo_cfg: Mapping[str, Any]) -> Demonstrator:
    actions = demo_cfg["actions"]
    fallback = demo_cfg.get("fallback", "repeat_last")
    return FixedSequenceDemonstrator(actions=actions, fallback=fallback)


def make_rl_demonstrator(
    model: ComputationalModel,
    *,
    bandit_cfg: Mapping[str, Any],
    demo_cfg: Mapping[str, Any],
) -> Demonstrator:
    # demo_cfg should include: {"params": {...}}
    params = demo_cfg["params"]
    return RLDemonstrator(model=model, params=params)
