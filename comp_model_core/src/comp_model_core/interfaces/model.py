"""comp_model_core.interfaces.model

Computational model interfaces.

A computational model defines:
- how an agent chooses actions (choice rule),
- how it updates latent variables given observations.

Models interact with a :class:`~comp_model_core.interfaces.block_runner.BlockRunner`
through the environment contract (:class:`~comp_model_core.spec.EnvironmentSpec`).

Notes
-----
Trial-level interface constraints (e.g., available actions, hidden/noisy outcome
observations, social observation channels) are handled by the generator/replayer
and the block runner. In particular, generators typically handle action masking /
renormalization when available actions vary by trial.

See Also
--------
comp_model_core.interfaces.block_runner.BlockRunner
comp_model_core.interfaces.block_runner.SocialObservation
comp_model_core.spec.EnvironmentSpec
comp_model_core.params.ParameterSchema
comp_model_core.requirements.ModelRequirements
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

import numpy as np

from ..params import ParameterSchema
from ..requirements import ModelRequirements
from ..spec import EnvironmentSpec
from .block_runner import SocialObservation


class ComputationalModel(ABC):
    """Base interface for computational models."""

    @classmethod
    def requirements(cls) -> ModelRequirements:
        """Declare task/data requirements for compatibility validation.

        Returns
        -------
        ModelRequirements
            Requirements such as "must observe demonstrator outcome at least once".
        """
        return ModelRequirements()

    @property
    @abstractmethod
    def param_schema(self) -> ParameterSchema:
        """Parameter schema describing names, bounds, and transforms."""
        ...

    @property
    def param_names(self) -> Sequence[str]:
        """Return parameter names in schema order."""
        return self.param_schema.names

    def get_params(self) -> dict[str, float]:
        """Return current model parameters as a flat dict.

        Returns
        -------
        dict[str, float]
            Mapping from parameter name to value.
        """
        return {name: float(getattr(self, name)) for name in self.param_schema.names}

    def set_params(
        self,
        params: Mapping[str, Any],
        *,
        strict: bool = True,
        check_bounds: bool = False,
    ) -> None:
        """Validate and set model parameters from a mapping.

        Parameters
        ----------
        params : Mapping[str, Any]
            Parameter mapping. Values will be cast to float after validation.
        strict : bool, optional
            If True, reject unknown keys.
        check_bounds : bool, optional
            If True, enforce schema bounds (after any schema transforms).

        Notes
        -----
        Validation is performed by :meth:`comp_model_core.params.ParameterSchema.validate`.
        """
        validated = self.param_schema.validate(params, strict=strict, check_bounds=check_bounds)
        for k, v in validated.items():
            setattr(self, k, float(v))

    def supports(self, spec: EnvironmentSpec) -> bool:
        """Return True if the model can be applied to the given environment spec.

        Parameters
        ----------
        spec : EnvironmentSpec
            Environment contract to check.

        Returns
        -------
        bool
            True if supported.

        Notes
        -----
        Override this if your model requires specific outcome types, state structure,
        or social channels.
        """
        return True

    @abstractmethod
    def reset_block(self, *, spec: EnvironmentSpec) -> None:
        """Reset model state at the start of a block.

        Parameters
        ----------
        spec : EnvironmentSpec
            Environment contract for the upcoming block.
        """
        ...

    @abstractmethod
    def action_probs(self, *, state: Any, spec: EnvironmentSpec) -> np.ndarray:
        """Return action probabilities for a given state.

        Parameters
        ----------
        state : Any
            Environment state/context identifier.
        spec : EnvironmentSpec
            Environment contract (e.g., n_actions).

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_actions,)`` containing probabilities that sum to 1.
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
        """Update latent variables given an observation.

        Parameters
        ----------
        state : Any
            Environment state/context identifier at the time of the action.
        action : int
            Action taken.
        outcome : float or None
            Observed outcome (may be None if feedback is hidden on that trial).
        spec : EnvironmentSpec
            Environment contract.
        info : Mapping[str, Any] or None, optional
            Optional metadata (e.g., observation-model parameters used by runner).
        rng : numpy.random.Generator or None, optional
            RNG for stochastic updates (if needed).
        """
        ...


class SocialComputationalModel(ComputationalModel):
    """Extension that supports social observations.

    Notes
    -----
    Social models can optionally implement :meth:`social_update`. The default
    implementation is a no-op.
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
        """Update latent variables given a social observation.

        Parameters
        ----------
        state : Any
            Environment state/context identifier at the time the social signal is received.
        social : SocialObservation
            Demonstrator choices/outcomes (true and/or observed).
        spec : EnvironmentSpec
            Environment contract.
        info : Mapping[str, Any] or None, optional
            Optional metadata.
        rng : numpy.random.Generator or None, optional
            RNG for stochastic updates.

        Notes
        -----
        Override in models that learn from demonstrator signals.
        """
        return
