import numpy as np
from scipy.special import softmax
from abc import ABC, abstractmethod
from .base import BaseSimulator


class BetaModelSoftMax(BaseSimulator):
    def __init__(self, lr_own, lr_partner, n_choices, beta):
        self.params = np.ones((n_choices, 2))
        self.lr_own = lr_own
        self.lr_partner = lr_partner
        self.beta = beta

    def make_choice(self):
        values = self.params[:, 1] / self.params.sum(axis=1)
        # Calculate the probability of each action using the softmax function.
        choice_prob = softmax(values * self.beta)
        # Randomly select an action based on its probability.
        return np.random.choice(len(self.params), p=choice_prob)

    def learn_from_own(self, choice: int, reward: int) -> None:
        self.params[choice, reward] += self.lr_own

    def learn_from_partner(self, choice: int, reward: float) -> None:
        self.params[choice, reward] += self.lr_partner
