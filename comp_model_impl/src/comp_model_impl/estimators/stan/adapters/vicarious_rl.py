"""Stan adapter for the Vicarious RL model.

This adapter maps :class:`comp_model_impl.models.vicarious_rl.vicarious_rl.Vicarious_RL`
to the corresponding Stan templates and provides prior requirements and
data-augmentation hooks.

Notes
-----
The Stan templates for this adapter are located at::

    comp_model_impl/estimators/stan/vicarious_rl/indiv_body.stan
    comp_model_impl/estimators/stan/vicarious_rl/hier_body.stan
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from comp_model_core.interfaces.model import ComputationalModel

from .base import StanAdapter, StanProgramRef
from ....models import Vicarious_RL


@dataclass(frozen=True, slots=True)
class VicariousRLStanAdapter(StanAdapter):
    """Adapter that maps :class:`Vicarious_RL` to Stan templates.

    Parameters
    ----------
    model : comp_model_core.interfaces.model.ComputationalModel
        Must be an instance of :class:`Vicarious_RL`.

    Examples
    --------
    >>> from comp_model_impl.models import Vicarious_RL
    >>> adapter = VicariousRLStanAdapter(model=Vicarious_RL())
    >>> adapter.program("indiv").key
    'vicarious_rl'
    """

    model: ComputationalModel

    def __post_init__(self) -> None:
        """Validate that the wrapped model is a Vicarious_RL instance.

        Raises
        ------
        TypeError
            If ``model`` is not a :class:`Vicarious_RL`.
        """
        if not isinstance(self.model, Vicarious_RL):
            raise TypeError(
                f"{self.__class__.__name__} requires {Vicarious_RL.__name__}, "
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
        return StanProgramRef(family=family, key="vicarious_rl", program_name=f"vicarious_rl_{family}")

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
            return ["alpha_o", "beta"]
        if family == "hier":
            return ["mu_alpha_o", "sd_alpha_o", "mu_beta", "sd_beta"]
        raise ValueError(f"Unknown family: {family!r}")

    def augment_subject_data(self, data: dict[str, Any]) -> None:
        """Add model-specific constants to subject-level Stan data.

        Parameters
        ----------
        data : dict[str, Any]
            Stan data dictionary to mutate in-place.

        Notes
        -----
        ``beta_lower`` and ``beta_upper`` define the admissible range for the
        inverse-temperature parameter ``beta`` in the Stan template. The lower
        bound prevents numerical issues near zero, and the upper bound is taken
        from ``model.beta_max``.
        """
        data["beta_lower"] = 1e-6
        data["beta_upper"] = float(getattr(self.model, "beta_max", 20.0))

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
        return ["alpha_o", "beta"]

    def population_var_names(self) -> Sequence[str]:
        """Names of population-level Stan variables to summarize.

        Returns
        -------
        Sequence[str]
            Stan variable names used for population-level summaries.
        """
        return [
            "alpha_o_pop",
            "beta_pop",
            "mu_alpha_o_hat",
            "sd_alpha_o_hat",
            "mu_beta_hat",
            "sd_beta_hat",
        ]
