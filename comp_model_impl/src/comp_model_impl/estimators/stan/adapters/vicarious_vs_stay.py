"""Stan adapter for the Vicarious + Value Shaping + Stay (Vicarious_VS_Stay) model.

This adapter maps
:class:`comp_model_impl.models.vicarious_vs_stay.vicarious_vs_stay.Vicarious_VS_Stay`
to Stan templates and injects model-specific constants into the Stan data dict.

Notes
-----
The Stan templates for this adapter are located at::

    comp_model_impl/estimators/stan/vicarious_vs_stay/indiv_body.stan
    comp_model_impl/estimators/stan/vicarious_vs_stay/hier_body.stan
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from comp_model_core.interfaces.model import ComputationalModel

from comp_model_impl.models.vicarious_vs_stay.vicarious_vs_stay import Vicarious_VS_Stay

from .base import StanAdapter, StanProgramRef


@dataclass(frozen=True, slots=True)
class VicariousVSStayStanAdapter(StanAdapter):
    """Adapter that maps :class:`Vicarious_VS_Stay` to Stan templates.

    Parameters
    ----------
    model : comp_model_core.interfaces.model.ComputationalModel
        Must be an instance of :class:`Vicarious_VS_Stay`.

    Examples
    --------
    >>> from comp_model_impl.models import Vicarious_VS_Stay
    >>> adapter = VicariousVSStayStanAdapter(model=Vicarious_VS_Stay())
    >>> adapter.program("indiv").key
    'vicarious_vs_stay'
    """

    model: ComputationalModel

    def __post_init__(self) -> None:
        """Validate that the wrapped model is a Vicarious_VS_Stay instance.

        Raises
        ------
        TypeError
            If ``model`` is not a :class:`Vicarious_VS_Stay`.
        """
        if not isinstance(self.model, Vicarious_VS_Stay):
            raise TypeError(
                f"{self.__class__.__name__} requires {Vicarious_VS_Stay.__name__}, "
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
        return StanProgramRef(family=family, key="vicarious_vs_stay", program_name=f"vicarious_vs_stay_{family}")

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
            return ["alpha_o", "alpha_a", "beta", "kappa"]
        if family == "hier":
            return [
                "mu_alpha_o",
                "sd_alpha_o",
                "mu_alpha_a",
                "sd_alpha_a",
                "mu_beta",
                "sd_beta",
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

        Notes
        -----
        - ``pseudo_reward`` is used for vicarious updates when demonstrator
          outcomes are observed.
        - ``beta_lower`` lower-bounds the inverse-temperature parameter
          for numerical stability and to avoid extreme softmax temperatures.
        - ``kappa_abs_max`` bounds the perseveration term ``kappa``.
        """
        data["pseudo_reward"] = float(getattr(self.model, "pseudo_reward", 1.0))
        data["beta_lower"] = 1e-6
        kappa_abs_max = getattr(self.model, "kappa_abs_max", None)
        if kappa_abs_max is None:
            kappa_abs_max = getattr(self.model, "kappa_max", 1.0)
        data["kappa_abs_max"] = float(kappa_abs_max)

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
        return ["alpha_o", "alpha_a", "beta", "kappa"]

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
            "beta_pop",
            "kappa_pop",
            "mu_alpha_o_hat",
            "sd_alpha_o_hat",
            "mu_alpha_a_hat",
            "sd_alpha_a_hat",
            "mu_beta_hat",
            "sd_beta_hat",
            "mu_kappa_hat",
            "sd_kappa_hat",
        ]
