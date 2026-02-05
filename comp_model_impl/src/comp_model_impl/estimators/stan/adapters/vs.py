"""Stan adapter for the VS (Value Shaping) model.

This adapter maps :class:`comp_model_impl.models.vs.vs.VS` to Stan templates
and injects model-specific constants into the Stan data dict.

Notes
-----
The Stan templates for this adapter are located at::

    comp_model_impl/estimators/stan/vs/indiv_body.stan
    comp_model_impl/estimators/stan/vs/hier_body.stan
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from comp_model_core.interfaces.model import ComputationalModel

from .base import StanAdapter, StanProgramRef
from ....models import VS


@dataclass(frozen=True, slots=True)
class VSStanAdapter(StanAdapter):
    """Adapter that maps :class:`VS` to Stan templates.

    Parameters
    ----------
    model : comp_model_core.interfaces.model.ComputationalModel
        Must be an instance of :class:`VS`.

    Notes
    -----
    Template key is ``"vs"`` which corresponds to the directory
    ``estimators/stan/vs`` that contains ``indiv_body.stan`` and ``hier_body.stan``.

    Examples
    --------
    >>> from comp_model_impl.models import VS
    >>> adapter = VSStanAdapter(model=VS())
    >>> adapter.program("hier").key
    'vs'
    """

    model: ComputationalModel

    def __post_init__(self) -> None:
        """Validate that the wrapped model is a VS instance.

        Raises
        ------
        TypeError
            If ``model`` is not a :class:`VS`.
        """
        if not isinstance(self.model, VS):
            raise TypeError(
                f"{self.__class__.__name__} requires {VS.__name__}, "
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
        return StanProgramRef(family=family, key="vs", program_name=f"vs_{family}")

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
            return ["alpha_p", "alpha_i", "beta", "kappa"]
        if family == "hier":
            return [
                "mu_alpha_p",
                "sd_alpha_p",
                "mu_alpha_i",
                "sd_alpha_i",
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
        - ``beta_lower``/``beta_upper`` bound the inverse-temperature parameter
          to avoid numerical issues and extreme softmax temperatures.
        - ``kappa_abs_max`` bounds the perseveration term ``kappa``.
        - ``pseudo_reward`` is the reward used for vicarious updates when
          demonstrator outcomes are observed.
        """
        data["beta_lower"] = 1e-6
        data["beta_upper"] = float(getattr(self.model, "beta_max"))
        data["kappa_abs_max"] = float(getattr(self.model, "kappa_abs_max"))
        data["pseudo_reward"] = float(getattr(self.model, "pseudo_reward"))

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
        return ["alpha_p", "alpha_i", "beta", "kappa"]

    def population_var_names(self) -> Sequence[str]:
        """Names of population-level Stan variables to summarize.

        Returns
        -------
        Sequence[str]
            Stan variable names used for population-level summaries.
        """
        return [
            "alpha_p_pop",
            "alpha_i_pop",
            "kappa_pop",
            "beta_pop",
            "mu_alpha_p_hat",
            "sd_alpha_p_hat",
            "mu_alpha_i_hat",
            "sd_alpha_i_hat",
            "mu_beta_hat",
            "sd_beta_hat",
            "mu_kappa_hat",
            "sd_kappa_hat",
        ]
