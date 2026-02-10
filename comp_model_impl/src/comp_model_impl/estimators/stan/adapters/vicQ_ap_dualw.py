"""Stan adapter for the VicQ_AP_DualW model.

This adapter maps
:class:`comp_model_impl.models.vicQ_ap_dualw.vicQ_ap_dualw.VicQ_AP_DualW`
to Stan templates and injects model-specific constants into the Stan data dict.

Notes
-----
The Stan templates for this adapter are located at::

    comp_model_impl/estimators/stan/vicQ_ap_dualw/indiv_body.stan
    comp_model_impl/estimators/stan/vicQ_ap_dualw/hier_body.stan
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from comp_model_core.interfaces.model import ComputationalModel

from comp_model_impl.models.vicQ_ap_dualw.vicQ_ap_dualw import VicQ_AP_DualW

from .base import StanAdapter, StanProgramRef


@dataclass(frozen=True, slots=True)
class VicQAPDualWStanAdapter(StanAdapter):
    """Adapter that maps :class:`VicQ_AP_DualW` to Stan templates.

    Parameters
    ----------
    model : comp_model_core.interfaces.model.ComputationalModel
        Must be an instance of :class:`VicQ_AP_DualW`.

    Examples
    --------
    >>> from comp_model_impl.models.vicQ_ap_dualw.vicQ_ap_dualw import VicQ_AP_DualW
    >>> adapter = VicQAPDualWStanAdapter(model=VicQ_AP_DualW())
    >>> adapter.program("indiv").key
    'vicQ_ap_dualw'
    """

    model: ComputationalModel

    def __post_init__(self) -> None:
        """Validate that the wrapped model is a VicQ_AP_DualW instance.

        Raises
        ------
        TypeError
            If ``model`` is not a :class:`VicQ_AP_DualW`.
        """
        if not isinstance(self.model, VicQ_AP_DualW):
            raise TypeError(
                f"{self.__class__.__name__} requires {VicQ_AP_DualW.__name__}, "
                f"got {type(self.model).__name__}"
            )

    def program(self, family: str) -> StanProgramRef:
        """Return the Stan program reference for the requested family.

        Parameters
        ----------
        family : {"indiv", "hier"}
            Program family identifier.

        Returns
        -------
        StanProgramRef
            Reference to the Stan template.
        """
        return StanProgramRef(family=family, key="vicQ_ap_dualw", program_name=f"vicQ_ap_dualw_{family}")

    def required_priors(self, family: str) -> Sequence[str]:
        """Return required prior names for the given family.

        Parameters
        ----------
        family : {"indiv", "hier"}
            Program family identifier.

        Returns
        -------
        Sequence[str]
            Prior names required by the template.

        Raises
        ------
        ValueError
            If ``family`` is not recognized.
        """
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
        """Add model-specific constants to subject-level Stan data.

        Parameters
        ----------
        data : dict[str, Any]
            Stan data dictionary to mutate in-place.
        """
        data["beta_lower"] = 1e-6
        data["beta_upper"] = float(getattr(self.model, "beta_max", 20.0))
        data["kappa_abs_max"] = float(getattr(self.model, "kappa_abs_max", 5.0))

    def augment_study_data(self, data: dict[str, Any]) -> None:
        """Add model-specific constants to hierarchical Stan data.

        Parameters
        ----------
        data : dict[str, Any]
            Stan data dictionary to mutate in-place.
        """
        self.augment_subject_data(data)

    def subject_param_names(self) -> Sequence[str]:
        """Names of subject-level Stan variables to summarize.

        Returns
        -------
        Sequence[str]
            Stan variable names used for subject-level summaries.
        """
        return ["alpha_o", "alpha_a", "beta_q", "beta_a", "kappa"]

    def population_var_names(self) -> Sequence[str]:
        """Names of population-level Stan variables to summarize.

        Returns
        -------
        Sequence[str]
            Stan variable names used for population-level summaries.
        """
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
