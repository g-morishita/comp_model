"""Stan adapter for the VicQ_AP_IndepDualW model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from comp_model_core.interfaces.model import ComputationalModel

from comp_model_impl.models.vicQ_ap_dualw_stay.vicQ_ap_indep_dualw import VicQ_AP_IndepDualW

from .base import StanAdapter, StanProgramRef


@dataclass(frozen=True, slots=True)
class VicQAPIndepDualWStanAdapter(StanAdapter):
    """Adapter that maps :class:`VicQ_AP_IndepDualW` to Stan templates."""

    model: ComputationalModel

    def __post_init__(self) -> None:
        if not isinstance(self.model, VicQ_AP_IndepDualW):
            raise TypeError(
                f"{self.__class__.__name__} requires {VicQ_AP_IndepDualW.__name__}, "
                f"got {type(self.model).__name__}"
            )

    def program(self, family: str) -> StanProgramRef:
        return StanProgramRef(
            family=family,
            key="vicQ_ap_indep_dualw",
            program_name=f"vicQ_ap_indep_dualw_{family}",
        )

    def required_priors(self, family: str) -> Sequence[str]:
        if family == "indiv":
            return ["alpha_o", "alpha_a", "beta_q", "beta_a", "kappa"]
        if family == "hier":
            return [
                "mu_alpha_o",
                "sd_alpha_o",
                "mu_alpha_a",
                "sd_alpha_a",
                "mu_beta_q",
                "sd_beta_q",
                "mu_beta_a",
                "sd_beta_a",
                "mu_kappa",
                "sd_kappa",
            ]
        raise ValueError(f"Unknown family: {family!r}")

    def augment_subject_data(self, data: dict[str, Any]) -> None:
        data["beta_lower"] = 1e-6
        data["kappa_abs_max"] = float(getattr(self.model, "kappa_abs_max", 5.0))

    def augment_study_data(self, data: dict[str, Any]) -> None:
        self.augment_subject_data(data)

    def subject_param_names(self) -> Sequence[str]:
        return ["alpha_o", "alpha_a", "beta_q", "beta_a", "kappa"]

    def population_var_names(self) -> Sequence[str]:
        return [
            "alpha_o_pop",
            "alpha_a_pop",
            "beta_q_pop",
            "beta_a_pop",
            "kappa_pop",
            "mu_alpha_o_hat",
            "sd_alpha_o_hat",
            "mu_alpha_a_hat",
            "sd_alpha_a_hat",
            "mu_beta_q_hat",
            "sd_beta_q_hat",
            "mu_beta_a_hat",
            "sd_beta_a_hat",
            "mu_kappa_hat",
            "sd_kappa_hat",
        ]
