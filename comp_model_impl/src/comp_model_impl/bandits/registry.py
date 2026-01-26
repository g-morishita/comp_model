from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from comp_model_core.interfaces.bandit import BanditEnv

BanditFactory = Callable[[Mapping[str, Any]], BanditEnv]


@dataclass(frozen=True, slots=True)
class BanditRegistry:
    factories: dict[str, BanditFactory]

    def make(self, bandit_type: str, cfg: Mapping[str, Any]) -> BanditEnv:
        try:
            f = self.factories[bandit_type]
        except KeyError as e:
            raise KeyError(f"Unknown bandit_type: {bandit_type}") from e
        return f(cfg)
