import numpy as np
from numpy import ndarray
from abc import ABC, abstractmethod
from typing import Sequence
from .arm import NormalArm, BernoulliArm


class Bandit(ABC):
    @abstractmethod
    def select_choice(self, choice: int) -> int | float:
        pass


class NormalBandit(Bandit):
    def __init__(
        self, means: Sequence[int | float], sds: Sequence[int | float]
    ) -> None:
        if len(means) != len(sds):
            raise ValueError(
                f"lengths of means and sds must match. \
                    len(means)={len(means)}, len(sds)={len(sds)}"
            )

        self.arms = []
        for mean, sd in zip(means, sds):
            self.arms.append(NormalArm(mean, sd))

    def select_choice(self, choice: int) -> float:
        if (choice < 0) or (choice >= len(self.arms)):
            raise ValueError(
                "chosen_arm must be between 0 and len(self.arms). \
                    {chosen_arm} is given."
            )

        return self.arms[choice].generate_reward()


class BernoulliBandit(Bandit):
    def __init__(
        self,
        means: Sequence[int | float],
    ) -> None:
        self.arms = []
        for mean in means:
            self.arms.append(BernoulliArm(mean))

    def select_choice(self, chosen_arm: int) -> float:
        if (chosen_arm < 0) or (chosen_arm >= len(self.arms)):
            raise ValueError(
                "chosen_arm must be between 0 and len(self.arms). \
                    {chosen_arm} is given."
            )

        return self.arms[chosen_arm].generate_reward()
