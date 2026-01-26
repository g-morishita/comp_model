"""
comp_model_core.interfaces.model

Computational model interfaces.

A computational model defines:

- how an agent chooses actions (choice rule),
- how it updates latent variables given observations.

Models interact with a :class:`~comp_model_core.interfaces.block_runner.BlockRunner`
through the environment contract (:class:`~comp_model_core.spec.EnvironmentSpec`).

Trial-level interface constraints (e.g., action availability) are handled by the
generator/replayer (masking/renormalization), not by the model itself.

Notes
-----
- Models declare plan/data compatibility constraints via
  :meth:`~ComputationalModel.requirements`.
- Parameters are validated and managed via :class:`~comp_model_core.params.ParameterSchema`.
- Social models may optionally implement :meth:`~SocialComputationalModel.social_update`
  to update from demonstrator observations.

See Also
--------
comp_model_core.spec.EnvironmentSpec
    Environment contract (number of actions, etc.).
comp_model_core.interfaces.block_runner.SocialObservation
    Social observation container for demonstrator choices/outcomes.
comp_model_core.requirements.Requirement
    Plan-based requirement protocol used for compatibility validation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

import numpy as np

from ..params import ParameterSchema
from ..requirements import Requirement
from ..spec import EnvironmentSpec
from .block_runner import SocialObservation


class ComputationalModel(ABC):
    """
    Base interface for computational models.

    A computational model encapsulates:

    - a **policy**: mapping from state/context to action probabilities, and
    - a **learning rule**: updating internal latent variables based on
      observations (e.g., outcomes, feedback).

    Parameters are stored as attributes on the instance and validated using the
    :class:`~comp_model_core.params.ParameterSchema` returned by
    :attr:`~ComputationalModel.param_schema`.

    Notes
    -----
    - Models operate against an :class:`~comp_model_core.spec.EnvironmentSpec`
      contract rather than concrete environment instances.
    - Trial-level action availability (forced-choice sets) should be handled by
      the generator/replayer by masking the probabilities returned by
      :meth:`~ComputationalModel.action_probs`.
    - Compatibility constraints about the task/data (e.g., needing outcome
      visibility) should be expressed via :meth:`~ComputationalModel.requirements`.

    See Also
    --------
    SocialComputationalModel
        Extension class for models that also update from social observations.
    comp_model_core.params.ParameterSchema
        Parameter validation/management.
    comp_model_core.requirements.Requirement
        Plan-based compatibility constraints.
    """

    @classmethod
    def requirements(cls) -> Sequence[Requirement]:
        """
        Declare plan/data requirements for compatibility validation.

        Returns
        -------
        Sequence[Requirement]
            Requirements that must be satisfied by a plan/environment for this
            model to be applicable. The default is an empty tuple (no additional
            requirements).

        Notes
        -----
        These are typically enforced by plan-based validation, not at runtime.
        """
        return ()

    @property
    @abstractmethod
    def param_schema(self) -> ParameterSchema:
        """
        Parameter schema for this model.

        Returns
        -------
        ParameterSchema
            Schema defining parameter names, types, and bounds.
        """
        ...

    @property
    def param_names(self) -> Sequence[str]:
        """
        Convenience accessor for parameter names.

        Returns
        -------
        Sequence[str]
            Names of parameters defined by :attr:`~ComputationalModel.param_schema`.
        """
        return self.param_schema.names

    def get_params(self) -> dict[str, float]:
        """
        Return current parameters as a plain dictionary.

        Returns
        -------
        dict[str, float]
            Mapping from parameter name to current float value.
        """
        return {name: float(getattr(self, name)) for name in self.param_schema.names}

    def set_params(
        self,
        params: Mapping[str, Any],
        *,
        strict: bool = True,
        check_bounds: bool = False,
    ) -> None:
        """
        Validate and set model parameters from a mapping.

        Parameters
        ----------
        params
            Mapping from parameter names to values.
        strict
            If True, unknown keys raise an error. If False, unknown keys are
            ignored.
        check_bounds
            If True, enforce declared parameter bounds during validation.

        Raises
        ------
        ValueError
            If validation fails (e.g., missing required keys, out-of-bounds
            values when ``check_bounds=True``).
        """
        validated = self.param_schema.validate(params, strict=strict, check_bounds=check_bounds)
        for k, v in validated.items():
            setattr(self, k, float(v))

    def supports(self, spec: EnvironmentSpec) -> bool:
        """
        Return True if the model can be applied to the given environment spec.

        Parameters
        ----------
        spec
            Environment contract.

        Returns
        -------
        bool
            True if the model supports the environment contract.

        Notes
        -----
        Subclasses may override to enforce constraints such as outcome type,
        number of actions, or state shape requirements.
        """
        return True

    @abstractmethod
    def reset_block(self, *, spec: EnvironmentSpec) -> None:
        """
        Reset model state at the start of a block.

        Parameters
        ----------
        spec
            Environment contract for the upcoming block.
        """
        ...

    @abstractmethod
    def action_probs(self, *, state: Any, spec: EnvironmentSpec) -> np.ndarray:
        """
        Compute action probabilities for the current trial.

        Parameters
        ----------
        state
            Current environment state/context identifier (opaque to the core API).
        spec
            Environment contract.

        Returns
        -------
        numpy.ndarray
            1D array of action probabilities with length ``spec.n_actions``.
            The probabilities should sum to 1 (before any generator masking).

        Notes
        -----
        Forced-choice action sets should be handled outside the model by masking
        and renormalizing these probabilities.
        """
        ...

    @abstractmethod
    def update(
        self,
        *,
        state: Any,
        action: int,
        outcome: float | None,
        spec: EnvironmentSpec,
        info: Mapping[str, Any] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """
        Update model latent state from an action and (possibly hidden) outcome.

        Parameters
        ----------
        state
            Current environment state/context identifier.
        action
            Action index that was taken.
        outcome
            Outcome as observed by the subject/model. May be ``None`` if feedback
            is hidden.
        spec
            Environment contract.
        info
            Optional metadata associated with the step (e.g., runner/environment
            info dict).
        rng
            Optional RNG for stochastic model updates.

        Notes
        -----
        This method should interpret ``outcome`` as the *observed* outcome (i.e.,
        after any visibility/noise transformations), not the true environment
        outcome.
        """
        ...


class SocialComputationalModel(ComputationalModel):
    """
    Extension that supports social observations.

    Social models may update their latent state from demonstrator information in
    addition to private outcomes.

    Notes
    -----
    The default implementation is a no-op. Subclasses can override
    :meth:`~SocialComputationalModel.social_update` to implement vicarious or
    imitation-based learning rules.
    """

    def social_update(
        self,
        *,
        state: Any,
        social: SocialObservation,
        spec: EnvironmentSpec,
        info: Mapping[str, Any] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """
        Update model latent state from a social observation.

        Parameters
        ----------
        state
            Current environment state/context identifier.
        social
            Social observation container (demonstrator choices/outcomes).
        spec
            Environment contract.
        info
            Optional metadata associated with the social observation.
        rng
            Optional RNG for stochastic model updates.

        Notes
        -----
        The default implementation performs no update.
        """
        return
