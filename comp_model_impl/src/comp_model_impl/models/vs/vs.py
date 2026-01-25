"""comp_model_impl.models.vs.vs

Value-Sharing (VS) model.

This is a social reinforcement-learning model that can incorporate demonstrator
information. The exact update equations depend on your implementation, but this
file typically defines:

- model parameters (via :class:`comp_model_core.params.ParameterSchema`)
- action selection rule (softmax or related)
- private outcome update (self outcomes)
- social update (demonstrator choices/outcomes)

Requirements
------------
This model may declare requirements via :meth:`requirements`, e.g., that it needs
demonstrator outcome visibility at least once.

See Also
--------
comp_model_core.interfaces.model.SocialComputationalModel
comp_model_core.requirements.ModelRequirements
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from comp_model_core.interfaces.model import SocialComputationalModel
from comp_model_core.params import ParameterSchema
from comp_model_core.requirements import ModelRequirements
from comp_model_core.spec import EnvironmentSpec


@dataclass(slots=True)
class VS(SocialComputationalModel):
    """A simple value-sharing social learning model.

    Notes
    -----
    This class skeleton assumes a standard Q-learning style update with an additional
    social update. Adapt the details to match your paper/model definition.
    """

    # Example parameters (ensure these match your actual schema/implementation)
    beta: float = 1.0
    alpha: float = 0.2
    w_social: float = 0.5

    def __post_init__(self) -> None:
        self._Q: np.ndarray | None = None

    @classmethod
    def requirements(cls) -> ModelRequirements:
        """Declare model requirements for compatibility validation."""
        # VS can often use demonstrator signals but may not strictly require them.
        return ModelRequirements()

    @property
    def param_schema(self) -> ParameterSchema:
        """Parameter schema for the VS model."""
        return ParameterSchema(
            names=("beta", "alpha", "w_social"),
            bounds={"beta": (1e-6, 100.0), "alpha": (0.0, 1.0), "w_social": (0.0, 1.0)},
        )

    def supports(self, spec: EnvironmentSpec) -> bool:
        """Return True if the model supports the given environment."""
        # VS typically assumes discrete actions; supports both social and asocial specs.
        return int(spec.n_actions) > 0

    def reset_block(self, *, spec: EnvironmentSpec) -> None:
        """Reset latent state for a new block.

        Parameters
        ----------
        spec : EnvironmentSpec
            Environment contract used to size internal arrays.
        """
        self._Q = np.zeros(int(spec.n_actions), dtype=float)

    def action_probs(self, *, state: Any, spec: EnvironmentSpec) -> np.ndarray:
        """Compute action probabilities via softmax.

        Parameters
        ----------
        state : Any
            Current state/context (unused for simple bandits).
        spec : EnvironmentSpec
            Environment contract.

        Returns
        -------
        numpy.ndarray
            Probability vector of shape ``(n_actions,)``.
        """
        assert self._Q is not None
        z = float(self.beta) * self._Q
        z = z - float(np.max(z))
        p = np.exp(z)
        return p / float(np.sum(p))

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
        """Update values from a private observed outcome.

        Parameters
        ----------
        state : Any
            State/context (unused in simple bandits).
        action : int
            Action taken.
        outcome : float or None
            Observed outcome. If None, this is a no-op.
        spec : EnvironmentSpec
            Environment contract.
        info : Mapping[str, Any] or None, optional
            Optional metadata.
        rng : numpy.random.Generator or None, optional
            RNG for stochastic updates (unused here).
        """
        if outcome is None:
            return
        assert self._Q is not None
        a = int(action)
        r = float(outcome)
        self._Q[a] = self._Q[a] + float(self.alpha) * (r - self._Q[a])

    def social_update(
        self,
        *,
        state: Any,
        social,
        spec: EnvironmentSpec,
        info: Mapping[str, Any] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Update from a social observation.

        Parameters
        ----------
        state : Any
            State/context (unused in simple bandits).
        social : SocialObservation
            Demonstrator observation.
        spec : EnvironmentSpec
            Environment contract.
        info : Mapping[str, Any] or None, optional
            Optional metadata.
        rng : numpy.random.Generator or None, optional
            RNG for stochastic updates (unused here).

        Notes
        -----
        This placeholder uses demonstrator *observed* outcomes when available.
        You may want to update from demonstrator choice alone (imitation) or
        combine choice and outcome signals differently.
        """
        if social is None or not social.others_choices:
            return
        assert self._Q is not None

        a = int(social.others_choices[0])

        # Use observed demonstrator outcome if present; otherwise do nothing.
        if social.observed_others_outcomes is None or len(social.observed_others_outcomes) == 0:
            return

        r = float(social.observed_others_outcomes[0])
        # Value-sharing: pull subject value toward demonstrator outcome signal.
        self._Q[a] = (1.0 - float(self.w_social)) * self._Q[a] + float(self.w_social) * r
