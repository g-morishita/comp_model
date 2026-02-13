"""Stan adapter for the MVS (mean-variance-skewness) model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from comp_model_core.interfaces.model import ComputationalModel

from .base import StanAdapter, StanProgramRef
from ....models import MVS


@dataclass(frozen=True, slots=True)
class MVSStanAdapter(StanAdapter):
    """Adapter that maps :class:`MVS` to Stan templates."""

    model: ComputationalModel

    def __post_init__(self) -> None:
        if not isinstance(self.model, MVS):
            raise TypeError(
                f"{self.__class__.__name__} requires {MVS.__name__}, "
                f"got {type(self.model).__name__}"
            )

    def program(self, family: str) -> StanProgramRef:
        return StanProgramRef(family=family, key="mvs", program_name=f"mvs_{family}")

    def required_priors(self, family: str) -> Sequence[str]:
        if family == "indiv":
            return ["lambda_var", "delta", "beta"]
        if family == "hier":
            return [
                "mu_lambda_var",
                "sd_lambda_var",
                "mu_delta",
                "sd_delta",
                "mu_beta",
                "sd_beta",
            ]
        raise ValueError(f"Unknown family: {family!r}")

    def augment_subject_data(self, data: dict[str, Any]) -> None:
        data["lambda_abs_max"] = float(getattr(self.model, "lambda_abs_max", 10.0))
        data["delta_abs_max"] = float(getattr(self.model, "delta_abs_max", 10.0))
        data["beta_lower"] = 1e-6
        data["beta_upper"] = float(getattr(self.model, "beta_max", 20.0))

    def augment_study_data(self, data: dict[str, Any]) -> None:
        self.augment_subject_data(data)

    def subject_param_names(self) -> Sequence[str]:
        return ["lambda_var", "delta", "beta"]

    def population_var_names(self) -> Sequence[str]:
        return [
            "lambda_var_pop",
            "delta_pop",
            "beta_pop",
            "mu_lambda_var_hat",
            "sd_lambda_var_hat",
            "mu_delta_hat",
            "sd_delta_hat",
            "mu_beta_hat",
            "sd_beta_hat",
        ]
