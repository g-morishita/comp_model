"""Stan adapter for the Vicarious DB-Stay model.

This adapter maps
:class:`comp_model_impl.models.vicarious_db_stay.vicarious_db_stay.Vicarious_DB_Stay`
to Stan templates and injects model-specific constants into the Stan data dict.

Notes
-----
The Stan templates for this adapter are located at::

    comp_model_impl/estimators/stan/vicarious_db_stay/indiv_body.stan
    comp_model_impl/estimators/stan/vicarious_db_stay/hier_body.stan
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from comp_model_core.interfaces.model import ComputationalModel

from comp_model_impl.models.vicarious_db_stay.vicarious_db_stay import Vicarious_DB_Stay

from .base import StanAdapter, StanProgramRef


@dataclass(frozen=True, slots=True)
class VicariousDBStayStanAdapter(StanAdapter):
    """Adapter that maps :class:`Vicarious_DB_Stay` to Stan templates.

    Parameters
    ----------
    model : comp_model_core.interfaces.model.ComputationalModel
        Must be an instance of :class:`Vicarious_DB_Stay`.

    Examples
    --------
    >>> from comp_model_impl.models import Vicarious_DB_Stay
    >>> adapter = VicariousDBStayStanAdapter(model=Vicarious_DB_Stay())
    >>> adapter.program("indiv").key
    'vicarious_db_stay'
    """

    model: ComputationalModel

    def __post_init__(self) -> None:
        """Validate that the wrapped model is a Vicarious_DB_Stay instance.

        Raises
        ------
        TypeError
            If ``model`` is not a :class:`Vicarious_DB_Stay`.
        """
        if not isinstance(self.model, Vicarious_DB_Stay):
            raise TypeError(
                f"{self.__class__.__name__} requires {Vicarious_DB_Stay.__name__}, "
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
        return StanProgramRef(family=family, key="vicarious_db_stay", program_name=f"vicarious_db_stay_{family}")

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
            return ["alpha_o", "demo_bias", "beta", "kappa"]
        if family == "hier":
            return [
                "mu_alpha_o",
                "sd_alpha_o",
                "mu_demo_bias",
                "sd_demo_bias",
                "mu_beta",
                "sd_beta",
                "mu_kappa",
                "sd_kappa",
            ]
        raise ValueError(f"Unknown family: {family!r}")

    def _abs_max_from_schema(self, name: str, default: float) -> float:
        """Infer an absolute bound from the model schema, if available."""
        schema = self.model.param_schema
        for p in schema.params:
            if p.name == name and p.bound is not None:
                lo = float(p.bound.lo)
                hi = float(p.bound.hi)
                return float(max(abs(lo), abs(hi)))
        return float(default)

    def augment_subject_data(self, data: dict[str, Any]) -> None:
        """Add model-specific constants to subject-level Stan data.

        Parameters
        ----------
        data : dict[str, Any]
            Stan data dictionary to mutate in-place.
        """
        data["beta_lower"] = 1e-6
        data["kappa_abs_max"] = float(getattr(self.model, "kappa_abs_max", 5.0))
        data["demo_bias_abs_max"] = self._abs_max_from_schema("demo_bias", 5.0)

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
        return ["alpha_o", "demo_bias", "beta", "kappa"]

    def population_var_names(self) -> Sequence[str]:
        """Names of population-level Stan variables to summarize.

        Returns
        -------
        Sequence[str]
            Stan variable names used for population-level summaries.
        """
        return [
            "alpha_o_pop",
            "demo_bias_pop",
            "beta_pop",
            "kappa_pop",
            "mu_alpha_o_hat",
            "sd_alpha_o_hat",
            "mu_demo_bias_hat",
            "sd_demo_bias_hat",
            "mu_beta_hat",
            "sd_beta_hat",
            "mu_kappa_hat",
            "sd_kappa_hat",
        ]
