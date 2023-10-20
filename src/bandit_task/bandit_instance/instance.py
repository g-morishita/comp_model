import numpy as np
from numpy import ndarray
from abc import ABC, abstractmethod
from typing import Sequence
from .arm import NormalArm, BernoulliArm


class Bandit(ABC):
    """
    Abstract base class representing a generic bandit with multiple arms.

    Methods
    -------
    select_choice(choice : int) -> int | float
        Abstract method to select and execute a choice (pull an arm) to get a reward.
    """

    def __init__(self):
        self.arms = None

    @abstractmethod
    def select_choice(self, choice: int) -> int | float:
        """Selects an arm (choice) and returns the corresponding reward."""
        pass


class NormalBandit(Bandit):
    """
    Represents a multi-armed bandit with arms that follow a normal distribution.

    Parameters
    ----------
    means : Sequence[int | float]
        List of means for each arm's normal distribution.
    sds : Sequence[int | float]
        List of standard deviations for each arm's normal distribution.
    """

    def __init__(
        self, means: Sequence[int | float], sds: Sequence[int | float]
    ) -> None:
        super().__init__()
        if len(means) != len(sds):
            raise ValueError(
                f"lengths of means and sds must match. \
                    len(means)={len(means)}, len(sds)={len(sds)}"
            )

        self.arms = [NormalArm(mean, sd) for mean, sd in zip(means, sds)]

    def select_choice(self, choice: int) -> float:
        """
        Selects a specific arm (choice) and returns the corresponding reward from a normal distribution.

        Parameters
        ----------
        choice : int
            The index of the arm to select.

        Returns
        -------
        float
            The reward value.
        """
        if (choice < 0) or (choice >= len(self.arms)):
            raise ValueError(
                f"choice must be between 0 and {len(self.arms) - 1}. \
                    {choice} is given."
            )

        return self.arms[choice].generate_reward()


class BernoulliBandit(Bandit):
    """
    Represents a multi-armed bandit with arms that follow a Bernoulli distribution.

    Parameters
    ----------
    means : Sequence[int | float]
        List of probabilities (mean) for each arm's Bernoulli distribution.
    """

    def __init__(self, means: Sequence[int | float]) -> None:
        super().__init__()
        self.arms = [BernoulliArm(mean) for mean in means]

    def select_choice(self, choice: int) -> int:
        """
        Selects a specific arm (choice) and returns the corresponding reward from a Bernoulli distribution.

        Parameters
        ----------
        choice : int
            The index of the arm to select.

        Returns
        -------
        int
            The reward value (0 or 1).
        """
        if (choice < 0) or (choice >= len(self.arms)):
            raise ValueError(
                f"choice must be between 0 and {len(self.arms) - 1}. \
                    {choice} is given."
            )

        return self.arms[choice].generate_reward()
