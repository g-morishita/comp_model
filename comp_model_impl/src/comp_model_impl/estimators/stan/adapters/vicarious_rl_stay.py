"""Stan adapter for the Vicarious_RL_Stay model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from comp_model_core.interfaces.model import ComputationalModel

from .base import StanAdapter, StanProgramRef
from ....models import Vicarious_RL_Stay


@dataclass(frozen=True, slots=True)
class VicariousRLStayStanAdapter(StanAdapter):
    """Adapter that maps :class:`Vicarious_RL_Stay` to Stan templates."""

    model: ComputationalModel

    def __post_init__(self) -> None:
        if not isinstance(self.model, Vicarious_RL_Stay):
            raise TypeError(
                f"{self.__class__.__name__} requires {Vicarious_RL_Stay.__name__}, "
                f"got {type(self.model).__name__}"
            )

    def program(self, family: str) -> StanProgramRef:
        return StanProgramRef(family=family, key="vicarious_rl_stay", program_name=f"vicarious_rl_stay_{family}")

    def required_priors(self, family: str) -> Sequence[str]:
        if family == "indiv":
            return ["alpha_o", "beta", "kappa"]
        if family == "hier":
            return ["mu_alpha_o", "sd_alpha_o", "mu_beta", "sd_beta", "mu_kappa", "sd_kappa"]
        raise ValueError(f"Unknown family: {family!r}")

    def augment_subject_data(self, data: dict[str, Any]) -> None:
        data["beta_lower"] = 1e-6
        data["beta_upper"] = float(getattr(self.model, "beta_max", 20.0))
        data["kappa_abs_max"] = float(getattr(self.model, "kappa_abs_max", 5.0))

    def augment_study_data(self, data: dict[str, Any]) -> None:
        self.augment_subject_data(data)

    def subject_param_names(self) -> Sequence[str]:
        return ["alpha_o", "beta", "kappa"]

    def population_var_names(self) -> Sequence[str]:
        return [
            "alpha_o_pop",
            "beta_pop",
            "kappa_pop",
            "mu_alpha_o_hat",
            "sd_alpha_o_hat",
            "mu_beta_hat",
            "sd_beta_hat",
            "mu_kappa_hat",
            "sd_kappa_hat",
        ]
