from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

import numpy as np

from ..spec import TaskSpec
from .bandit import SocialObservation
from ..params import ParameterSchema

class ComputationalModel(ABC):
    """
    Base class for computational models.

    Subclasses should implement:
    - param_schema
    - supports(spec)
    - reset_block(spec=...)
    - action_probs(...)
    - update(...)
    """

    @property
    @abstractmethod
    def param_schema(self) -> ParameterSchema:
        ...

    @property
    def param_names(self) -> Sequence[str]:
        return self.param_schema.names
    
    def get_params(self) -> dict[str, float]:
        """Return current parameters (by schema names)."""
        return {name: float(getattr(self, name)) for name in self.param_schema.names}

    def set_params(
        self,
        params: Mapping[str, Any],
        *,
        strict: bool = True,
        check_bounds: bool = False,
    ) -> None:
        """Safely set model parameters using the schema."""
        validated = self.param_schema.validate(
            params,
            strict=strict,
            check_bounds=check_bounds,
        )
        for k, v in validated.items():
            setattr(self, k, float(v))

    def supports(self, spec: TaskSpec) -> bool:
        return True

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
        outcome: float,
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
