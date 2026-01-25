"""comp_model_impl.models.vicarious_rl.vicarious_rl

Vicarious reinforcement learning model.

This model learns not only from the subject's own outcomes, but also from observed
demonstrator outcomes. Therefore, the model typically requires that demonstrator
outcomes are observed at least once within a block (or dataset).

Compatibility requirements
--------------------------
This implementation declares a requirement that *at least one* trial provides a
non-hidden demonstrator outcome observation. Compatibility validation is performed
by :func:`comp_model_core.validation.validate_runner_against_model_requirements`.

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
class Vicarious_RL(SocialComputationalModel):
    """A simple vicarious reinforcement learning model (placeholder implementation)."""

    beta: float = 1.0
    alpha_self: float = 0.2
    alpha_demo: float = 0.2

    def __post_init__(self) -> None:
        self._Q: np.ndarray | None = None

    @classmethod
    def requirements(cls) -> ModelRequirements:
        """Declare requirements for task/data compatibility."""
        return ModelRequirements(needs_demo_outcome_at_least_once=True)

    @property
    def param_schema(self) -> ParameterSchema:
        """Parameter schema for vicarious RL."""
        return ParameterSchema(
            names=("beta", "alpha_self", "alpha_demo"),
            bounds={
                "beta": (1e-6, 100.0),
                "alpha_self": (0.0, 1.0),
                "alpha_demo": (0.0, 1.0),
            },
        )

    def supports(self, spec: EnvironmentSpec) -> bool:
        """Return True if the model supports the environment."""
        # Requires a social environment spec.
        return bool(spec.is_social) and int(spec.n_actions) > 0

    def reset_block(self, *, spec: EnvironmentSpec) -> None:
        """Reset latent state for a new block."""
        self._Q = np.zeros(int(spec.n_actions), dtype=float)

    def action_probs(self, *, state: Any, spec: EnvironmentSpec) -> np.ndarray:
        """Compute softmax action probabilities."""
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
        """Update from subject's own observed outcome."""
        if outcome is None:
            return
        assert self._Q is not None
        a = int(action)
        r = float(outcome)
        self._Q[a] = self._Q[a] + float(self.alpha_self) * (r - self._Q[a])

    def social_update(
        self,
        *,
        state: Any,
        social,
        spec: EnvironmentSpec,
        info: Mapping[str, Any] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Update from demonstrator observation.

        Notes
        -----
        This uses the demonstrator's *observed* outcome (as seen by the subject),
        which may be noisy or hidden depending on trial specs.
        """
        if social is None or not social.others_choices:
            return
        assert self._Q is not None

        a = int(social.others_choices[0])
        if social.observed_others_outcomes is None or len(social.observed_others_outcomes) == 0:
            return
        r = float(social.observed_others_outcomes[0])
        self._Q[a] = self._Q[a] + float(self.alpha_demo) * (r - self._Q[a])
