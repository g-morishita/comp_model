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
        # Calculate the probability of each action using the softmax function.
        choice_prob = softmax(values * self.beta)
        # Randomly select an action based on its probability.
        return np.random.choice(len(self.params), p=choice_prob)

    def learn(self, choice: int, reward: int) -> None:
        self.params[choice, reward] += self.lr


class BetaModelInfoBonusSoftmax(BetaModelSoftMax):
    def __init__(self, n_choices, lr, beta, coef_info_bonus):
        super().__init__(n_choices, lr, beta)
        self.coef_info_bonus = coef_info_bonus

    def make_choice(self):
        sum_params = self.params.sum(axis=1)
        product_params = self.params.prod(axis=1)
        variance = product_params / ((sum_params**2) * (sum_params + 1))
        values = self.params[:, 1] / self.params.sum(
            axis=1
        ) + self.coef_info_bonus * np.std(variance)
        # Calculate the probability of each action using the softmax function.
        choice_prob = softmax(values * self.beta)
        # Randomly select an action based on its probability.
        return np.random.choice(len(self.params), p=choice_prob)
