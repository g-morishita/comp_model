from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

import numpy as np

from ..spec import TaskSpec
from .bandit import SocialObservation


class ComputationalModel(ABC):
    """Parameters are typically subject-level; latents reset per block."""

    @property
    @abstractmethod
    def param_names(self) -> Sequence[str]:
        ...

    def supports(self, spec: TaskSpec) -> bool:
        return True

    def set_params(self, params: Mapping[str, float]) -> None:
        """Default injection: set attributes if they exist."""
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)

    @abstractmethod
    def reset_block(self, *, spec: TaskSpec) -> None:
        """Reset latent state at the beginning of a block."""
        ...

    @abstractmethod
    def action_probs(self, *, state: Any, spec: TaskSpec) -> np.ndarray:
        ...

    @abstractmethod
    def update(
        self,
        *,
        state: Any,
        action: int,
        reward: float,
        spec: TaskSpec,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        ...


class SocialComputationalModel(ComputationalModel):
    def social_update(
        self,
        *,
        state: Any,
        social: SocialObservation,
        spec: TaskSpec,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        return
