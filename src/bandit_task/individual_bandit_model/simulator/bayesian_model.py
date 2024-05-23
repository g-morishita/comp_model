import numpy as np
from scipy.special import softmax
from .base import BaseSimulator


class BetaModelSoftMax(BaseSimulator):
    def __init__(self, n_choices, lr, beta):
        super().__init__()
        self.params = np.ones((n_choices, 2))
        self.lr = lr
        self.beta = beta

    def make_choice(self):
        values = self.params[:, 1] / self.params.sum(axis=1)
        print(values)
        # Calculate the probability of each action using the softmax function.
        choice_prob = softmax(values * self.beta)
        # Randomly select an action based on its probability.
        return np.random.choice(len(self.params), p=choice_prob)

    def learn(self, choice: int, reward: int) -> None:
        print(self.params)
        self.params[choice, reward] += self.lr
