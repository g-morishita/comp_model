"""Stan adapter for within-subject (shared+delta) VS models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from comp_model_core.interfaces.model import ComputationalModel

from .base import StanAdapter, StanProgramRef
from ....models import VS
from ....models.within_subject_shared_delta import (
    ConditionedSharedDeltaModel,
    ConditionedSharedDeltaSocialModel,
)


@dataclass(frozen=True, slots=True)
class VSWithinSubjectStanAdapter(StanAdapter):
    """Adapter for :class:`~comp_model_impl.models.VS` wrapped in a shared+delta conditioner."""

    model: ComputationalModel

    def __post_init__(self) -> None:
        if not isinstance(self.model, (ConditionedSharedDeltaModel, ConditionedSharedDeltaSocialModel)):
            raise TypeError(
                f"{self.__class__.__name__} requires a ConditionedSharedDelta*Model wrapper, "
                f"got {type(self.model).__name__}"
            )
        base = getattr(self.model, "base_model", None)
        if not isinstance(base, VS):
            raise TypeError(
                f"{self.__class__.__name__} requires base_model={VS.__name__}, got {type(base).__name__}"
            )

    @property
    def base_model(self) -> VS:
        return getattr(self.model, "base_model")

    @property
    def conditions(self) -> Sequence[str]:
        return list(getattr(self.model, "conditions"))

    @property
    def baseline_condition(self) -> str:
        return str(getattr(self.model, "baseline_condition"))

    def program(self, family: str) -> StanProgramRef:
        return StanProgramRef(
            family=family,
            key="vs_within_subject",
            program_name=f"vs_within_subject_{family}",
        )

    def required_priors(self, family: str) -> Sequence[str]:
        if family == "indiv":
            return [
                "alpha_p_shared",
                "alpha_p_delta",
                "alpha_i_shared",
                "alpha_i_delta",
                "beta_shared",
                "beta_delta",
                "kappa_shared",
                "kappa_delta",
            ]

        if family == "hier":
            return [
                # shared
                "mu_ap_shared",
                "sd_ap_shared",
                "mu_ai_shared",
                "sd_ai_shared",
                "mu_b_shared",
                "sd_b_shared",
                "mu_k_shared",
                "sd_k_shared",
                # delta (applies elementwise across C-1 conditions)
                "mu_ap_delta",
                "sd_ap_delta",
                "mu_ai_delta",
                "sd_ai_delta",
                "mu_b_delta",
                "sd_b_delta",
                "mu_k_delta",
                "sd_k_delta",
            ]

        raise ValueError(f"Unknown family: {family!r}")

    def augment_subject_data(self, data: dict[str, Any]) -> None:
        # mirror VSStanAdapter
        beta_max = float(getattr(self.base_model, "beta_max"))
        data["beta_lower"] = float(1e-6)
        data["beta_upper"] = float(beta_max)
        data["kappa_abs_max"] = float(getattr(self.base_model, "kappa_abs_max"))
        data["pseudo_reward"] = float(getattr(self.base_model, "pseudo_reward"))

    def augment_study_data(self, data: dict[str, Any]) -> None:
        return None

    def subject_param_names(self) -> Sequence[str]:
        # variable names in generated quantities
        return [
            "alpha_p_hat",
            "alpha_i_hat",
            "beta_hat",
            "kappa_hat",
            "alpha_p_shared_z_hat",
            "alpha_i_shared_z_hat",
            "beta_shared_z_hat",
            "kappa_shared_z_hat",
            "alpha_p_delta_z_hat",
            "alpha_i_delta_z_hat",
            "beta_delta_z_hat",
            "kappa_delta_z_hat",
        ]

    def population_var_names(self) -> Sequence[str]:
        return [
            "alpha_p_pop",
            "alpha_i_pop",
            "beta_pop",
            "kappa_pop",
            "mu_ap_shared_hat",
            "sd_ap_shared_hat",
            "mu_ai_shared_hat",
            "sd_ai_shared_hat",
            "mu_b_shared_hat",
            "sd_b_shared_hat",
            "mu_k_shared_hat",
            "sd_k_shared_hat",
            "mu_ap_delta_hat",
            "sd_ap_delta_hat",
            "mu_ai_delta_hat",
            "sd_ai_delta_hat",
            "mu_b_delta_hat",
            "sd_b_delta_hat",
            "mu_k_delta_hat",
            "sd_k_delta_hat",
        ]
