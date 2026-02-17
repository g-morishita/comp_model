"""Stan adapter for within-subject (shared+delta) Vicarious_RL_Stay models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from comp_model_core.interfaces.model import ComputationalModel

from .base import StanAdapter, StanProgramRef
from ....models import Vicarious_RL_Stay
from ....models.within_subject_shared_delta import (
    ConditionedSharedDeltaModel,
    ConditionedSharedDeltaSocialModel,
)


@dataclass(frozen=True, slots=True)
class VicariousRLStayWithinSubjectStanAdapter(StanAdapter):
    """Adapter for :class:`Vicarious_RL_Stay` wrapped in shared+delta conditioner."""

    model: ComputationalModel

    def __post_init__(self) -> None:
        if not isinstance(self.model, (ConditionedSharedDeltaModel, ConditionedSharedDeltaSocialModel)):
            raise TypeError(
                f"{self.__class__.__name__} requires a ConditionedSharedDelta*Model wrapper, "
                f"got {type(self.model).__name__}"
            )
        base = getattr(self.model, "base_model", None)
        if not isinstance(base, Vicarious_RL_Stay):
            raise TypeError(
                f"{self.__class__.__name__} requires base_model={Vicarious_RL_Stay.__name__}, got {type(base).__name__}"
            )

    @property
    def base_model(self) -> Vicarious_RL_Stay:
        return getattr(self.model, "base_model")

    def program(self, family: str) -> StanProgramRef:
        return StanProgramRef(
            family=family,
            key="vicarious_rl_stay_within_subject",
            program_name=f"vicarious_rl_stay_within_subject_{family}",
        )

    def required_priors(self, family: str) -> Sequence[str]:
        if family == "indiv":
            return [
                "alpha_o__shared",
                "alpha_o__delta",
                "beta__shared",
                "beta__delta",
                "kappa__shared",
                "kappa__delta",
            ]

        if family == "hier":
            return [
                "mu_alpha_o__shared",
                "sd_alpha_o__shared",
                "mu_beta__shared",
                "sd_beta__shared",
                "mu_kappa__shared",
                "sd_kappa__shared",
                "mu_alpha_o__delta",
                "sd_alpha_o__delta",
                "mu_beta__delta",
                "sd_beta__delta",
                "mu_kappa__delta",
                "sd_kappa__delta",
            ]

        raise ValueError(f"Unknown family: {family!r}")

    def augment_subject_data(self, data: dict[str, Any]) -> None:
        data["beta_lower"] = float(1e-6)
        data["kappa_abs_max"] = float(getattr(self.base_model, "kappa_abs_max"))

    def augment_study_data(self, data: dict[str, Any]) -> None:
        self.augment_subject_data(data)

    def subject_param_names(self) -> Sequence[str]:
        return [
            "alpha_o_hat",
            "beta_hat",
            "kappa_hat",
            "alpha_o__shared_z_hat",
            "beta__shared_z_hat",
            "kappa__shared_z_hat",
            "alpha_o__delta_z_hat",
            "beta__delta_z_hat",
            "kappa__delta_z_hat",
        ]

    def population_var_names(self) -> Sequence[str]:
        return [
            "alpha_o_pop",
            "beta_pop",
            "kappa_pop",
            "mu_alpha_o__shared_hat",
            "sd_alpha_o__shared_hat",
            "mu_beta__shared_hat",
            "sd_beta__shared_hat",
            "mu_kappa__shared_hat",
            "sd_kappa__shared_hat",
            "mu_alpha_o__delta_hat",
            "sd_alpha_o__delta_hat",
            "mu_beta__delta_hat",
            "sd_beta__delta_hat",
            "mu_kappa__delta_hat",
            "sd_kappa__delta_hat",
        ]
