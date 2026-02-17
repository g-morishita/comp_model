"""Stan adapter for within-subject (shared+delta) VS models.

This adapter targets :class:`VS` models wrapped with the shared+delta
within-subject conditioner and maps them to the corresponding Stan templates.

Notes
-----
The Stan templates for this adapter are located at::

    comp_model_impl/estimators/stan/vs_within_subject/indiv_body.stan
    comp_model_impl/estimators/stan/vs_within_subject/hier_body.stan
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from comp_model_core.interfaces.model import ComputationalModel

from .base import StanAdapter, StanProgramRef
from ....models import VS
from ....models.within_subject_shared_delta import (
    ConditionedSharedDeltaModel,
    ConditionedSharedDeltaSocialModel,
)


@dataclass(frozen=True, slots=True)
class VSWithinSubjectStanAdapter(StanAdapter):
    """Adapter for :class:`VS` wrapped in a shared+delta conditioner.

    Parameters
    ----------
    model : comp_model_core.interfaces.model.ComputationalModel
        Must be a shared+delta wrapper around :class:`VS`.

    Examples
    --------
    >>> from comp_model_impl.models import VS
    >>> from comp_model_impl.models.within_subject_shared_delta import wrap_model_with_shared_delta_conditions
    >>> wrapped = wrap_model_with_shared_delta_conditions(
    ...     model=VS(),
    ...     conditions=["A", "B"],
    ...     baseline_condition="A",
    ... )
    >>> adapter = VSWithinSubjectStanAdapter(model=wrapped)
    >>> adapter.program("indiv").key
    'vs_within_subject'
    """

    model: ComputationalModel

    def __post_init__(self) -> None:
        """Validate wrapper type and base model.

        Raises
        ------
        TypeError
            If ``model`` is not a shared+delta wrapper or the base model is not
            :class:`VS`.
        """
        if not isinstance(self.model, (ConditionedSharedDeltaModel, ConditionedSharedDeltaSocialModel)):
            raise TypeError(
                f"{self.__class__.__name__} requires a ConditionedSharedDelta*Model wrapper, "
                f"got {type(self.model).__name__}"
            )
        base = getattr(self.model, "base_model", None)
        if not isinstance(base, VS):
            raise TypeError(
                f"{self.__class__.__name__} requires base_model={VS.__name__}, got {type(base).__name__}"
            )

    @property
    def base_model(self) -> VS:
        """Return the wrapped base model.

        Returns
        -------
        VS
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
            key="vs_within_subject",
            program_name=f"vs_within_subject_{family}",
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
                "alpha_p__shared",
                "alpha_p__delta",
                "alpha_i__shared",
                "alpha_i__delta",
                "beta__shared",
                "beta__delta",
                "kappa__shared",
                "kappa__delta",
            ]

        if family == "hier":
            return [
                # shared
                "mu_alpha_p__shared",
                "sd_alpha_p__shared",
                "mu_alpha_i__shared",
                "sd_alpha_i__shared",
                "mu_beta__shared",
                "sd_beta__shared",
                "mu_kappa__shared",
                "sd_kappa__shared",
                # delta (applies elementwise across C-1 conditions)
                "mu_alpha_p__delta",
                "sd_alpha_p__delta",
                "mu_alpha_i__delta",
                "sd_alpha_i__delta",
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

        Notes
        -----
        Mirrors :class:`VSStanAdapter`:
        - ``beta_lower`` lower-bounds the inverse-temperature parameter.
        - ``kappa_abs_max`` bounds the perseveration term ``kappa``.
        - ``pseudo_reward`` is used for vicarious updates when demonstrator
          outcomes are observed.
        """
        data["beta_lower"] = float(1e-6)
        data["kappa_abs_max"] = float(getattr(self.base_model, "kappa_abs_max"))
        data["pseudo_reward"] = float(getattr(self.base_model, "pseudo_reward"))

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
            "alpha_p_hat",
            "alpha_i_hat",
            "beta_hat",
            "kappa_hat",
            "alpha_p__shared_z_hat",
            "alpha_i__shared_z_hat",
            "beta__shared_z_hat",
            "kappa__shared_z_hat",
            "alpha_p__delta_z_hat",
            "alpha_i__delta_z_hat",
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
            "alpha_p__pop",
            "alpha_i__pop",
            "beta__pop",
            "kappa__pop",
            "mu_alpha_p__shared_hat",
            "sd_alpha_p__shared_hat",
            "mu_alpha_i__shared_hat",
            "sd_alpha_i__shared_hat",
            "mu_beta__shared_hat",
            "sd_beta__shared_hat",
            "mu_kappa__shared_hat",
            "sd_kappa__shared_hat",
            "mu_alpha_p__delta_hat",
            "sd_alpha_p__delta_hat",
            "mu_alpha_i__delta_hat",
            "sd_alpha_i__delta_hat",
            "mu_beta__delta_hat",
            "sd_beta__delta_hat",
            "mu_kappa__delta_hat",
            "sd_kappa__delta_hat",
        ]
