"""Stationary multi-armed bandit problem implementation.

The class in this module is intentionally a concrete problem implementation,
not the library's core abstraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from comp_model_v2.core.contracts import DecisionContext
from comp_model_v2.plugins import ComponentManifest


@dataclass(frozen=True, slots=True)
class BanditOutcome:
    """Outcome emitted by :class:`StationaryBanditProblem`.

    Parameters
    ----------
    reward : float
        Sampled scalar reward.
    reward_probability : float
        Bernoulli success probability associated with the chosen arm.
    """

    reward: float
    reward_probability: float


class StationaryBanditProblem:
    """Bernoulli bandit with optional per-trial action availability.

    Parameters
    ----------
    reward_probabilities : Sequence[float]
        Bernoulli reward probabilities for each arm. Arm IDs are integer
        indices ``0..n_arms-1``.
    action_schedule : Sequence[Sequence[int]] | None, optional
        Optional action availability schedule. If provided, its length defines
        the valid trial range and each trial returns the corresponding action
        subset.

    Raises
    ------
    ValueError
        If probabilities are out of bounds, no arms are provided, or schedule
        entries contain invalid/empty action sets.

    Notes
    -----
    This class is stateless across trials except for initialization checks.
    It demonstrates how a task-specific environment can fit the generic core
    protocols.
    """

    def __init__(
        self,
        reward_probabilities: Sequence[float],
        action_schedule: Sequence[Sequence[int]] | None = None,
    ) -> None:
        if len(reward_probabilities) == 0:
            raise ValueError("reward_probabilities must contain at least one arm")

        self._reward_probabilities = tuple(float(p) for p in reward_probabilities)
        for p in self._reward_probabilities:
            if p < 0.0 or p > 1.0:
                raise ValueError("reward probabilities must be between 0 and 1")

        self._all_actions = tuple(range(len(self._reward_probabilities)))

        if action_schedule is None:
            self._action_schedule = None
        else:
            normalized_schedule: list[tuple[int, ...]] = []
            for trial_actions in action_schedule:
                actions = tuple(int(a) for a in trial_actions)
                if len(actions) == 0:
                    raise ValueError("each scheduled trial must allow at least one action")
                if not set(actions).issubset(set(self._all_actions)):
                    raise ValueError("action_schedule contains out-of-range action indices")
                normalized_schedule.append(actions)
            self._action_schedule = tuple(normalized_schedule)

    def reset(self, *, rng: np.random.Generator) -> None:
        """Reset state before an episode.

        Parameters
        ----------
        rng : numpy.random.Generator
            Runtime RNG. Unused by this stationary implementation.
        """

    def available_actions(self, *, trial_index: int) -> tuple[int, ...]:
        """Return the legal arm IDs for the trial.

        Parameters
        ----------
        trial_index : int
            Zero-based trial index.

        Returns
        -------
        tuple[int, ...]
            Available arm IDs.

        Raises
        ------
        IndexError
            If a schedule exists and ``trial_index`` exceeds it.
        """

        if self._action_schedule is None:
            return self._all_actions

        return self._action_schedule[trial_index]

    def observe(self, *, context: DecisionContext[int]) -> dict[str, int]:
        """Return a minimal observation containing the trial index.

        Parameters
        ----------
        context : DecisionContext[int]
            Per-trial context.

        Returns
        -------
        dict[str, int]
            Observation payload.
        """

        return {"trial_index": context.trial_index}

    def transition(
        self,
        action: int,
        *,
        context: DecisionContext[int],
        rng: np.random.Generator,
    ) -> BanditOutcome:
        """Execute one arm pull and return sampled outcome.

        Parameters
        ----------
        action : int
            Chosen arm ID.
        context : DecisionContext[int]
            Per-trial context.
        rng : numpy.random.Generator
            Random generator used for Bernoulli sampling.

        Returns
        -------
        BanditOutcome
            Reward sample and associated probability.

        Raises
        ------
        ValueError
            If ``action`` is not legal for this trial.
        """

        if action not in context.available_actions:
            raise ValueError(f"action {action!r} is not available on trial {context.trial_index}")

        p_reward = self._reward_probabilities[action]
        reward = float(rng.random() < p_reward)
        return BanditOutcome(reward=reward, reward_probability=p_reward)


def create_stationary_bandit_problem(
    *,
    reward_probabilities: Sequence[float],
    action_schedule: Sequence[Sequence[int]] | None = None,
) -> StationaryBanditProblem:
    """Factory used by plugin discovery.

    Parameters
    ----------
    reward_probabilities : Sequence[float]
        Bernoulli reward probabilities for each arm.
    action_schedule : Sequence[Sequence[int]] | None, optional
        Optional per-trial available action schedule.

    Returns
    -------
    StationaryBanditProblem
        Configured problem instance.
    """

    return StationaryBanditProblem(
        reward_probabilities=reward_probabilities,
        action_schedule=action_schedule,
    )


PLUGIN_MANIFESTS = [
    ComponentManifest(
        kind="problem",
        component_id="stationary_bandit",
        factory=create_stationary_bandit_problem,
        description="Stationary Bernoulli multi-armed bandit",
    )
]
