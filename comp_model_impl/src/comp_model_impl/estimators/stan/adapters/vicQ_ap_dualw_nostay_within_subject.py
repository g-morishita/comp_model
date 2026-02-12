"""Stan adapter for within-subject (shared+delta) VicQ_AP_DualW_NoStay models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from comp_model_core.interfaces.model import ComputationalModel

from .base import StanAdapter, StanProgramRef
from ....models import VicQ_AP_DualW_NoStay
from ....models.within_subject_shared_delta import (
    ConditionedSharedDeltaModel,
    ConditionedSharedDeltaSocialModel,
)


@dataclass(frozen=True, slots=True)
class VicQAPDualWNoStayWithinSubjectStanAdapter(StanAdapter):
    """Adapter for :class:`VicQ_AP_DualW_NoStay` wrapped in shared+delta conditioner."""

    model: ComputationalModel

    def __post_init__(self) -> None:
        if not isinstance(self.model, (ConditionedSharedDeltaModel, ConditionedSharedDeltaSocialModel)):
            raise TypeError(
                f"{self.__class__.__name__} requires a ConditionedSharedDelta*Model wrapper, "
                f"got {type(self.model).__name__}"
            )
        base = getattr(self.model, "base_model", None)
        if not isinstance(base, VicQ_AP_DualW_NoStay):
            raise TypeError(
                f"{self.__class__.__name__} requires base_model={VicQ_AP_DualW_NoStay.__name__}, "
                f"got {type(base).__name__}"
            )

    @property
    def base_model(self) -> VicQ_AP_DualW_NoStay:
        return getattr(self.model, "base_model")

    def program(self, family: str) -> StanProgramRef:
        return StanProgramRef(
            family=family,
            key="vicQ_ap_dualw_nostay_within_subject",
            program_name=f"vicQ_ap_dualw_nostay_within_subject_{family}",
        )

    def required_priors(self, family: str) -> Sequence[str]:
        if family == "indiv":
            return [
                "alpha_o__shared",
                "alpha_o__delta",
                "alpha_a__shared",
                "alpha_a__delta",
                "beta__shared",
                "beta__delta",
                "w__shared",
                "w__delta",
            ]

        if family == "hier":
            return [
                "mu_alpha_o__shared",
                "sd_alpha_o__shared",
                "mu_alpha_a__shared",
                "sd_alpha_a__shared",
                "mu_beta__shared",
                "sd_beta__shared",
                "mu_w__shared",
                "sd_w__shared",
                "mu_alpha_o__delta",
                "sd_alpha_o__delta",
                "mu_alpha_a__delta",
                "sd_alpha_a__delta",
                "mu_beta__delta",
                "sd_beta__delta",
                "mu_w__delta",
                "sd_w__delta",
            ]

        raise ValueError(f"Unknown family: {family!r}")

    def augment_subject_data(self, data: dict[str, Any]) -> None:
        data["beta_lower"] = float(1e-6)
        data["beta_upper"] = float(getattr(self.base_model, "beta_max"))

    def augment_study_data(self, data: dict[str, Any]) -> None:
        self.augment_subject_data(data)

    def subject_param_names(self) -> Sequence[str]:
        return [
            "alpha_o_hat",
            "alpha_a_hat",
            "beta_hat",
            "w_hat",
            "alpha_o__shared_z_hat",
            "alpha_a__shared_z_hat",
            "beta__shared_z_hat",
            "w__shared_z_hat",
            "alpha_o__delta_z_hat",
            "alpha_a__delta_z_hat",
            "beta__delta_z_hat",
            "w__delta_z_hat",
        ]

    def population_var_names(self) -> Sequence[str]:
        return [
            "alpha_o_pop",
            "alpha_a_pop",
            "beta_pop",
            "w_pop",
            "mu_alpha_o__shared_hat",
            "sd_alpha_o__shared_hat",
            "mu_alpha_a__shared_hat",
            "sd_alpha_a__shared_hat",
            "mu_beta__shared_hat",
            "sd_beta__shared_hat",
            "mu_w__shared_hat",
            "sd_w__shared_hat",
            "mu_alpha_o__delta_hat",
            "sd_alpha_o__delta_hat",
            "mu_alpha_a__delta_hat",
            "sd_alpha_a__delta_hat",
            "mu_beta__delta_hat",
            "sd_beta__delta_hat",
            "mu_w__delta_hat",
            "sd_w__delta_hat",
        ]
