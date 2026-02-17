"""Stan adapter for within-subject (shared+delta) Vicarious_DB_Stay models.

This adapter targets :class:`Vicarious_DB_Stay` models wrapped with the shared+delta
within-subject conditioner and maps them to the corresponding Stan templates.

Notes
-----
The Stan templates for this adapter are located at::

    comp_model_impl/estimators/stan/vicarious_db_stay_within_subject/indiv_body.stan
    comp_model_impl/estimators/stan/vicarious_db_stay_within_subject/hier_body.stan
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from comp_model_core.interfaces.model import ComputationalModel

from .base import StanAdapter, StanProgramRef
from ....models import Vicarious_DB_Stay
from ....models.within_subject_shared_delta import (
    ConditionedSharedDeltaModel,
    ConditionedSharedDeltaSocialModel,
)


@dataclass(frozen=True, slots=True)
class VicariousDBStayWithinSubjectStanAdapter(StanAdapter):
    """Adapter for :class:`Vicarious_DB_Stay` wrapped in a shared+delta conditioner.

    Parameters
    ----------
    model : comp_model_core.interfaces.model.ComputationalModel
        Must be a shared+delta wrapper around :class:`Vicarious_DB_Stay`.

    Examples
    --------
    >>> from comp_model_impl.models import Vicarious_DB_Stay
    >>> from comp_model_impl.models.within_subject_shared_delta import wrap_model_with_shared_delta_conditions
    >>> wrapped = wrap_model_with_shared_delta_conditions(
    ...     model=Vicarious_DB_Stay(),
    ...     conditions=["A", "B"],
    ...     baseline_condition="A",
    ... )
    >>> adapter = VicariousDBStayWithinSubjectStanAdapter(model=wrapped)
    >>> adapter.program("indiv").key
    'vicarious_db_stay_within_subject'
    """

    model: ComputationalModel

    def __post_init__(self) -> None:
        """Validate wrapper type and base model.

        Raises
        ------
        TypeError
            If ``model`` is not a shared+delta wrapper or the base model is not
            :class:`Vicarious_DB_Stay`.
        """
        if not isinstance(self.model, (ConditionedSharedDeltaModel, ConditionedSharedDeltaSocialModel)):
            raise TypeError(
                f"{self.__class__.__name__} requires a ConditionedSharedDelta*Model wrapper, "
                f"got {type(self.model).__name__}"
            )
        base = getattr(self.model, "base_model", None)
        if not isinstance(base, Vicarious_DB_Stay):
            raise TypeError(
                f"{self.__class__.__name__} requires base_model={Vicarious_DB_Stay.__name__}, got {type(base).__name__}"
            )

    @property
    def base_model(self) -> Vicarious_DB_Stay:
        """Return the wrapped base model.

        Returns
        -------
        Vicarious_DB_Stay
            Base model instance.
        """
        return getattr(self.model, "base_model")

    @property
    def conditions(self) -> Sequence[str]:
        """Return ordered condition labels from the wrapper.

        Returns
        -------
        Sequence[str]
            Condition labels in the wrapper-defined order.
        """
        return list(getattr(self.model, "conditions"))

    @property
    def baseline_condition(self) -> str:
        """Return the baseline condition label.

        Returns
        -------
        str
            Baseline condition label.
        """
        return str(getattr(self.model, "baseline_condition"))

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
        return StanProgramRef(
            family=family,
            key="vicarious_db_stay_within_subject",
            program_name=f"vicarious_db_stay_within_subject_{family}",
        )

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
            return [
                "alpha_o__shared",
                "alpha_o__delta",
                "demo_bias__shared",
                "demo_bias__delta",
                "beta__shared",
                "beta__delta",
                "kappa__shared",
                "kappa__delta",
            ]

        if family == "hier":
            return [
                # shared
                "mu_alpha_o__shared",
                "sd_alpha_o__shared",
                "mu_demo_bias__shared",
                "sd_demo_bias__shared",
                "mu_beta__shared",
                "sd_beta__shared",
                "mu_kappa__shared",
                "sd_kappa__shared",
                # delta (applies elementwise across C-1 conditions)
                "mu_alpha_o__delta",
                "sd_alpha_o__delta",
                "mu_demo_bias__delta",
                "sd_demo_bias__delta",
                "mu_beta__delta",
                "sd_beta__delta",
                "mu_kappa__delta",
                "sd_kappa__delta",
            ]

        raise ValueError(f"Unknown family: {family!r}")

    def augment_subject_data(self, data: dict[str, Any]) -> None:
        """Add model-specific constants to subject-level Stan data.

        Parameters
        ----------
        data : dict[str, Any]
            Stan data dictionary to mutate in-place.
        """
        data["beta_lower"] = float(1e-6)
        data["kappa_abs_max"] = float(getattr(self.base_model, "kappa_abs_max"))
        data["demo_bias_abs_max"] = float(getattr(self.base_model, "demo_bias_abs_max"))

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
        return [
            "alpha_o_hat",
            "demo_bias_hat",
            "beta_hat",
            "kappa_hat",
            "alpha_o__shared_z_hat",
            "demo_bias__shared_z_hat",
            "beta__shared_z_hat",
            "kappa__shared_z_hat",
            "alpha_o__delta_z_hat",
            "demo_bias__delta_z_hat",
            "beta__delta_z_hat",
            "kappa__delta_z_hat",
        ]

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
            "mu_alpha_o__shared_hat",
            "sd_alpha_o__shared_hat",
            "mu_demo_bias__shared_hat",
            "sd_demo_bias__shared_hat",
            "mu_beta__shared_hat",
            "sd_beta__shared_hat",
            "mu_kappa__shared_hat",
            "sd_kappa__shared_hat",
            "mu_alpha_o__delta_hat",
            "sd_alpha_o__delta_hat",
            "mu_demo_bias__delta_hat",
            "sd_demo_bias__delta_hat",
            "mu_beta__delta_hat",
            "sd_beta__delta_hat",
            "mu_kappa__delta_hat",
            "sd_kappa__delta_hat",
        ]
