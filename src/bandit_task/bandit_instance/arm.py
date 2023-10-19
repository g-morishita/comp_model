from abc import ABC

import numpy as np


class Arm(ABC):
    """
    Abstract base class representing an arm in a multi-armed bandit setup.

    Methods
    -------
    generate_reward()
        Abstract method to be implemented by subclasses to generate reward values.
    """

    def generate_reward(self) -> int | float:
        """
        Generate a reward after selecting this arm.

        Returns
        -------
        Union[int, float]
            The reward value.
        """
        pass


class NormalArm(Arm):
    """
    Represents an arm with normally distributed rewards.

    Parameters
    ----------
    mean : Union[int, float]
        The mean of the normal distribution.
    sd : Union[int, float]
        The standard deviation of the normal distribution.
    """

    def __init__(self, mean: int | float, sd: int | float) -> None:
        self.mean = mean
        self.sd = sd

    def generate_reward(self) -> float:
        """
        Generate a reward from a normal distribution based on the given mean and sd.

        Returns
        -------
        float
            The reward value.
        """
        return np.random.normal(self.mean, self.sd)


class BernoulliArm(Arm):
    """
    Represents an arm with Bernoulli-distributed rewards.

    Parameters
    ----------
    mean : Union[int, float]
        The probability parameter (p) for the Bernoulli distribution.

    Raises
    ------
    ValueError
        If the provided mean is not between 0 and 1.
    """

    def __init__(self, mean: int | float) -> None:
        if (mean < 0) or (mean > 1):
            raise ValueError(f"mean must be between 0 and 1. {mean} is given.")
        self.mean = mean

    def generate_reward(self) -> int:
        """
        Generate a reward from a Bernoulli distribution with the given probability parameter.

        Returns
        -------
        int
            The reward value (0 or 1).
        """
        return np.random.binomial(n=1, p=self.mean)
