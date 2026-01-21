from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from comp_model_core.interfaces.demonstrator import Demonstrator

DemonstratorFactory = Callable[[Mapping[str, Any], Mapping[str, Any]], Demonstrator]
#                           ^ bandit_cfg        ^ demo_cfg


@dataclass(frozen=True, slots=True)
class DemonstratorRegistry:
    factories: dict[str, DemonstratorFactory]

    def make(
        self,
        demonstrator_type: str,
        *,
        bandit_cfg: Mapping[str, Any],
        demo_cfg: Mapping[str, Any],
    ) -> Demonstrator:
        try:
            f = self.factories[demonstrator_type]
        except KeyError as e:
            raise KeyError(f"Unknown demonstrator_type: {demonstrator_type}") from e
        return f(bandit_cfg=bandit_cfg, demo_cfg=demo_cfg)
