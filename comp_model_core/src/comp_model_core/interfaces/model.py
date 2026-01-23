"""
Computational model interfaces.

A computational model defines how an agent chooses actions and updates latent state
given observations from a task environment.

This module defines:

- :class:`ComputationalModel` for asocial tasks.
- :class:`SocialComputationalModel` for tasks with social observations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

import numpy as np

from ..spec import TaskSpec
from .bandit import SocialObservation
from ..params import ParameterSchema


class ComputationalModel(ABC):
    """
    Abstract base class for computational models.

    Subclasses should implement:
    - :attr:`param_schema`
    - :meth:`supports`
    - :meth:`reset_block`
    - :meth:`action_probs`
    - :meth:`update`

    Notes
    -----
    Models are stateful: they hold latent variables (e.g., Q-values). Generators and
    estimators should reset state at each block boundary via :meth:`reset_block`.
    """

    @property
    @abstractmethod
    def param_schema(self) -> ParameterSchema:
        """
        Return the parameter schema for this model.

        Returns
        -------
        ParameterSchema
            Schema defining parameter names, defaults, bounds, and transforms.
        """
        ...

    @property
    def param_names(self) -> Sequence[str]:
        """
        Return parameter names defined by :attr:`param_schema`.

        Returns
        -------
        Sequence[str]
            Parameter names in schema order.
        """
        return self.param_schema.names

    def get_params(self) -> dict[str, float]:
        """
        Return current parameter values.

        Returns
        -------
        dict[str, float]
            Mapping from parameter name to current value.
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
        Set model parameters using the schema for validation/coercion.

        Parameters
        ----------
        params : Mapping[str, Any]
            Parameter values keyed by name.
        strict : bool, optional
            If ``True``, unknown keys raise an error. If ``False``, unknown keys are
            ignored.
        check_bounds : bool, optional
            If ``True``, validate values against declared bounds.

        Raises
        ------
        comp_model_core.errors.ParameterValidationError
            If values cannot be coerced to finite floats.
        ValueError
            If unknown keys are present and ``strict=True``.
        """
        validated = self.param_schema.validate(
            params,
            strict=strict,
            check_bounds=check_bounds,
        )
        for k, v in validated.items():
            setattr(self, k, float(v))

    def supports(self, spec: TaskSpec) -> bool:
        """
        Check whether the model supports a given task specification.

        Parameters
        ----------
        spec : TaskSpec
            Task specification to check.

        Returns
        -------
        bool
            ``True`` if supported.

        Notes
        -----
        The base implementation returns ``True``. Override this to enforce constraints
        (e.g., require ``spec.is_social`` or a particular ``outcome_type``).
        """
        return True

    @abstractmethod
    def reset_block(self, *, spec: TaskSpec) -> None:
        """
        Reset latent state at the beginning of a block.

        Parameters
        ----------
        spec : TaskSpec
            Task specification for the block.
        """
        ...

    @abstractmethod
    def action_probs(self, *, state: Any, spec: TaskSpec) -> np.ndarray:
        """
        Compute action probabilities for a state.

        Parameters
        ----------
        state : Any
            State/context identifier.
        spec : TaskSpec
            Task specification.

        Returns
        -------
        numpy.ndarray
            Probability vector of shape ``(spec.n_actions,)`` that sums to 1.
        """
        ...

    @abstractmethod
    def update(
        self,
        *,
        state: Any,
        action: int,
        outcome: float | None,
        spec: TaskSpec,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Update latent state after observing an outcome.

        Parameters
        ----------
        state : Any
            State/context identifier for the trial.
        action : int
            Action taken.
        outcome : float or None
            Outcome observed by the agent/subject. ``None`` indicates hidden outcome.
        spec : TaskSpec
            Task specification.
        info : Mapping[str, Any] or None, optional
            Additional task-specific information.

        Notes
        -----
        Generators/estimators define the timing/order of calling this method.
        """
        ...


class SocialComputationalModel(ComputationalModel):
    """
    Extension of :class:`ComputationalModel` that supports social observations.
    """

    def social_update(
        self,
        *,
        state: Any,
        social: SocialObservation,
        spec: TaskSpec,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Update latent state after observing social information.

        Parameters
        ----------
        state : Any
            State/context identifier for the trial.
        social : SocialObservation
            Social observation payload (demonstrator actions/outcomes).
        spec : TaskSpec
            Task specification.
        info : Mapping[str, Any] or None, optional
            Additional task-specific information.

        Notes
        -----
        The base implementation is a no-op. Social-learning models should override this.
        """
        return
