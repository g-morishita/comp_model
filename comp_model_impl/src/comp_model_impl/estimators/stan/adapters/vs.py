"""Stan adapter for the VS (Value Shaping) model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from comp_model_core.interfaces.model import ComputationalModel

from .base import StanAdapter, StanProgramRef
from ....models import VS


@dataclass(frozen=True, slots=True)
class VSStanAdapter(StanAdapter):
    """Adapter that maps :class:`~comp_model_impl.models.vs.vs.VS` to Stan templates.

    Notes
    -----
    Template key is ``"vs"`` which corresponds to the directory
    ``estimators/stan/vs`` that contains ``indiv_body.stan`` and ``hier_body.stan``.
    """

    model: ComputationalModel

    def __post_init__(self) -> None:
        if not isinstance(self.model, VS):
            raise TypeError(
                f"{self.__class__.__name__} requires {VS.__name__}, "
                f"got {type(self.model).__name__}"
            )

    def program(self, family: str) -> StanProgramRef:
        return StanProgramRef(family=family, key="vs", program_name=f"vs_{family}")

    def required_priors(self, family: str) -> Sequence[str]:
        if family == "indiv":
            return ["alpha_p", "alpha_i", "beta", "kappa"]
        if family == "hier":
            return [
                "mu_ap",
                "sd_ap",
                "mu_ai",
                "sd_ai",
                "mu_b",
                "sd_b",
                "mu_k",
                "sd_k",
            ]
        raise ValueError(f"Unknown family: {family!r}")

    def augment_subject_data(self, data: dict[str, Any]) -> None:
        data["beta_lower"] = 1e-6
        data["beta_upper"] = float(getattr(self.model, "beta_max"))
        data["kappa_abs_max"] = float(getattr(self.model, "kappa_abs_max"))
        data["pseudo_reward"] = float(getattr(self.model, "pseudo_reward"))

    def augment_study_data(self, data: dict[str, Any]) -> None:
        # Same constants as in the subject program.
        self.augment_subject_data(data)

    def subject_param_names(self) -> Sequence[str]:
        return ["alpha_p", "alpha_i", "beta", "kappa"]

    def population_var_names(self) -> Sequence[str]:
        return [
            "alpha_p_pop",
            "alpha_i_pop",
            "kappa_pop",
            "beta_pop",
            "mu_ap_hat",
            "sd_ap_hat",
            "mu_ai_hat",
            "sd_ai_hat",
            "mu_b_hat",
            "sd_b_hat",
            "mu_k_hat",
            "sd_k_hat",
        ]
