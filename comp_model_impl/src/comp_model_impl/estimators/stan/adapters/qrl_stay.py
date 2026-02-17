"""Stan adapter for the QRL_Stay model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from comp_model_core.interfaces.model import ComputationalModel

from ....models import QRL_Stay
from .base import StanAdapter, StanProgramRef


@dataclass(frozen=True, slots=True)
class QRLStayStanAdapter(StanAdapter):
    """Adapter that maps :class:`QRL_Stay` to Stan templates."""

    model: ComputationalModel

    def __post_init__(self) -> None:
        if not isinstance(self.model, QRL_Stay):
            raise TypeError(
                f"{self.__class__.__name__} requires {QRL_Stay.__name__}, "
                f"got {type(self.model).__name__}"
            )

    def program(self, family: str) -> StanProgramRef:
        return StanProgramRef(
            family=family,
            key="qrl_stay",
            program_name=f"qrl_stay_{family}",
        )

    def required_priors(self, family: str) -> Sequence[str]:
        if family == "indiv":
            return ["alpha", "beta", "kappa"]
        if family == "hier":
            return [
                "mu_alpha",
                "sd_alpha",
                "mu_beta",
                "sd_beta",
                "mu_kappa",
                "sd_kappa",
            ]
        raise ValueError(f"Unknown family: {family!r}")

    def augment_subject_data(self, data: dict[str, Any]) -> None:
        data["beta_lower"] = 1e-6

    def augment_study_data(self, data: dict[str, Any]) -> None:
        self.augment_subject_data(data)

    def subject_param_names(self) -> Sequence[str]:
        return ["alpha", "beta", "kappa"]

    def population_var_names(self) -> Sequence[str]:
        return [
            "alpha_pop",
            "beta_pop",
            "kappa_pop",
            "mu_alpha_hat",
            "sd_alpha_hat",
            "mu_beta_hat",
            "sd_beta_hat",
            "mu_kappa_hat",
            "sd_kappa_hat",
        ]
