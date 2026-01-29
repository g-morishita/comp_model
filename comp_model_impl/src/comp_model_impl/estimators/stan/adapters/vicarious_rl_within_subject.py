"""Stan adapter for within-subject (shared+delta) VicariousRL models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from comp_model_core.interfaces.model import ComputationalModel

from .base import StanAdapter, StanProgramRef
from ....models import Vicarious_RL
from ....models.within_subject_shared_delta import (
    ConditionedSharedDeltaModel,
    ConditionedSharedDeltaSocialModel,
)


@dataclass(frozen=True, slots=True)
class VicariousRLWithinSubjectStanAdapter(StanAdapter):
    """Adapter for :class:`~comp_model_impl.models.Vicarious_RL` wrapped in shared+delta conditioner."""

    model: ComputationalModel

    def __post_init__(self) -> None:
        if not isinstance(self.model, (ConditionedSharedDeltaModel, ConditionedSharedDeltaSocialModel)):
            raise TypeError(
                f"{self.__class__.__name__} requires a ConditionedSharedDelta*Model wrapper, "
                f"got {type(self.model).__name__}"
            )
        base = getattr(self.model, "base_model", None)
        if not isinstance(base, Vicarious_RL):
            raise TypeError(
                f"{self.__class__.__name__} requires base_model={Vicarious_RL.__name__}, got {type(base).__name__}"
            )

    @property
    def base_model(self) -> Vicarious_RL:
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
            key="vicarious_rl_within_subject",
            program_name=f"vicarious_rl_within_subject_{family}",
        )

    def required_priors(self, family: str) -> Sequence[str]:
        if family == "indiv":
            return [
                "alpha_o_shared",
                "alpha_o_delta",
                "beta_shared",
                "beta_delta",
            ]

        if family == "hier":
            return [
                "mu_ao_shared",
                "sd_ao_shared",
                "mu_b_shared",
                "sd_b_shared",
                "mu_ao_delta",
                "sd_ao_delta",
                "mu_b_delta",
                "sd_b_delta",
            ]

        raise ValueError(f"Unknown family: {family!r}")

    def augment_subject_data(self, data: dict[str, Any]) -> None:
        beta_max = float(getattr(self.base_model, "beta_max"))
        data["beta_lower"] = float(1e-6)
        data["beta_upper"] = float(beta_max)

    def augment_study_data(self, data: dict[str, Any]) -> None:
        return None

    def subject_param_names(self) -> Sequence[str]:
        return [
            "alpha_o_hat",
            "beta_hat",
            "alpha_o_shared_z_hat",
            "beta_shared_z_hat",
            "alpha_o_delta_z_hat",
            "beta_delta_z_hat",
        ]

    def population_var_names(self) -> Sequence[str]:
        return [
            "alpha_o_pop",
            "beta_pop",
            "mu_ao_shared_hat",
            "sd_ao_shared_hat",
            "mu_b_shared_hat",
            "sd_b_shared_hat",
            "mu_ao_delta_hat",
            "sd_ao_delta_hat",
            "mu_b_delta_hat",
            "sd_b_delta_hat",
        ]
