"""Stan adapter for the VicQ_AP_DualW_NoStay model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from comp_model_core.interfaces.model import ComputationalModel

from comp_model_impl.models.vicQ_ap_dualw_nostay.vicQ_ap_dualw_nostay import VicQ_AP_DualW_NoStay

from .base import StanAdapter, StanProgramRef


@dataclass(frozen=True, slots=True)
class VicQAPDualWNoStayStanAdapter(StanAdapter):
    """Adapter that maps :class:`VicQ_AP_DualW_NoStay` to Stan templates."""

    model: ComputationalModel

    def __post_init__(self) -> None:
        if not isinstance(self.model, VicQ_AP_DualW_NoStay):
            raise TypeError(
                f"{self.__class__.__name__} requires {VicQ_AP_DualW_NoStay.__name__}, "
                f"got {type(self.model).__name__}"
            )

    def program(self, family: str) -> StanProgramRef:
        return StanProgramRef(
            family=family,
            key="vicQ_ap_dualw_nostay",
            program_name=f"vicQ_ap_dualw_nostay_{family}",
        )

    def required_priors(self, family: str) -> Sequence[str]:
        if family == "indiv":
            return ["alpha_o", "alpha_a", "beta", "w"]
        if family == "hier":
            return [
                "mu_alpha_o",
                "sd_alpha_o",
                "mu_alpha_a",
                "sd_alpha_a",
                "mu_beta",
                "sd_beta",
                "mu_w",
                "sd_w",
            ]
        raise ValueError(f"Unknown family: {family!r}")

    def augment_subject_data(self, data: dict[str, Any]) -> None:
        data["beta_lower"] = 1e-6
        data["beta_upper"] = float(getattr(self.model, "beta_max", 20.0))

    def augment_study_data(self, data: dict[str, Any]) -> None:
        self.augment_subject_data(data)

    def subject_param_names(self) -> Sequence[str]:
        return ["alpha_o", "alpha_a", "beta", "w"]

    def population_var_names(self) -> Sequence[str]:
        return [
            "alpha_o_pop",
            "alpha_a_pop",
            "beta_pop",
            "w_pop",
            "mu_alpha_o_hat",
            "sd_alpha_o_hat",
            "mu_alpha_a_hat",
            "sd_alpha_a_hat",
            "mu_beta_hat",
            "sd_beta_hat",
            "mu_w_hat",
            "sd_w_hat",
        ]
