"""Stan adapter for the QRL (Q-learning) model.

This adapter maps :class:`comp_model_impl.models.qrl.qrl.QRL` to Stan templates
and injects model-specific constants into the Stan data dict.

Notes
-----
The Stan templates for this adapter are located at::

    comp_model_impl/estimators/stan/qrl/indiv_body.stan
    comp_model_impl/estimators/stan/qrl/hier_body.stan
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from comp_model_core.interfaces.model import ComputationalModel

from .base import StanAdapter, StanProgramRef
from ....models import QRL


@dataclass(frozen=True, slots=True)
class QRLStanAdapter(StanAdapter):
    """Adapter that maps :class:`QRL` to Stan templates.

    Parameters
    ----------
    model : comp_model_core.interfaces.model.ComputationalModel
        Must be an instance of :class:`QRL`.

    Examples
    --------
    >>> from comp_model_impl.models import QRL
    >>> adapter = QRLStanAdapter(model=QRL())
    >>> adapter.program("indiv").key
    'qrl'
    """

    model: ComputationalModel

    def __post_init__(self) -> None:
        """Validate that the wrapped model is a QRL instance.

        Raises
        ------
        TypeError
            If ``model`` is not a :class:`QRL`.
        """
        if not isinstance(self.model, QRL):
            raise TypeError(
                f"{self.__class__.__name__} requires {QRL.__name__}, "
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
        return StanProgramRef(family=family, key="qrl", program_name=f"qrl_{family}")

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
            return ["alpha", "beta"]
        if family == "hier":
            return ["mu_alpha", "sd_alpha", "mu_beta", "sd_beta"]
        raise ValueError(f"Unknown family: {family!r}")

    def augment_subject_data(self, data: dict[str, Any]) -> None:
        """Add model-specific constants to subject-level Stan data.

        Parameters
        ----------
        data : dict[str, Any]
            Stan data dictionary to mutate in-place.

        Notes
        -----
        ``beta_lower`` defines the lower bound for the
        inverse-temperature parameter ``beta`` in the Stan template.
        """
        data["beta_lower"] = 1e-6

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
        return ["alpha", "beta"]

    def population_var_names(self) -> Sequence[str]:
        """Names of population-level Stan variables to summarize.

        Returns
        -------
        Sequence[str]
            Stan variable names used for population-level summaries.
        """
        return [
            "alpha_pop",
            "beta_pop",
            "mu_alpha_hat",
            "sd_alpha_hat",
            "mu_beta_hat",
            "sd_beta_hat",
        ]
