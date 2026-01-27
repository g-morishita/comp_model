"""Stan adapter for the Vicarious RL model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from comp_model_core.interfaces.model import ComputationalModel

from .base import StanAdapter, StanProgramRef
from ....models import Vicarious_RL


@dataclass(frozen=True, slots=True)
class VicariousRLStanAdapter(StanAdapter):
    """Adapter that maps :class:`~comp_model_impl.models.vicarious_rl.vicarious_rl.Vicarious_RL` to Stan templates."""

    model: ComputationalModel

    def __post_init__(self) -> None:
        if not isinstance(self.model, Vicarious_RL):
            raise TypeError(
                f"{self.__class__.__name__} requires {Vicarious_RL.__name__}, "
                f"got {type(self.model).__name__}"
            )

    def program(self, family: str) -> StanProgramRef:
        return StanProgramRef(family=family, key="vicarious_rl", program_name=f"vicarious_rl_{family}")

    def required_priors(self, family: str) -> Sequence[str]:
        if family == "indiv":
            return ["alpha_o", "beta"]
        if family == "hier":
            return ["mu_ao", "sd_ao", "mu_b", "sd_b"]
        raise ValueError(f"Unknown family: {family!r}")

    def augment_subject_data(self, data: dict[str, Any]) -> None:
        data["beta_lower"] = 1e-6
        data["beta_upper"] = float(getattr(self.model, "beta_max", 20.0))

    def augment_study_data(self, data: dict[str, Any]) -> None:
        self.augment_subject_data(data)

    def subject_param_names(self) -> Sequence[str]:
        return ["alpha_o", "beta"]

    def population_var_names(self) -> Sequence[str]:
        return ["alpha_o_pop", "beta_pop", "mu_ao_hat", "sd_ao_hat", "mu_b_hat", "sd_b_hat"]
