import numpy as np
from scipy.optimize import LinearConstraint
from scipy.special import softmax
from typing import Sequence

from .base import MLEstimator
from ...type import NDArrayNumber


class BetaModelSoftmaxMLEWithFixedIncrement(MLEstimator):
    def __init__(self):
        super().__init__()

    def neg_ll(self, params: Sequence[int | float]) -> float:
        beta = params[0]

        # Initialize the parameters with ones
        model_params = np.ones((len(self.choices), self.num_choices, 2))

        for t in range(1, len(self.choices)):
            model_params[t, self.choices[t - 1], self.rewards[t - 1]] += 1

            # For actions not taken, the parameter values remain the same
            for a in range(self.num_choices):
                if a != self.choices[t - 1]:
                    model_params[t, a, :] += model_params[t - 1, a, :]

        # Calculate choice probabilities using softmax function
        values = model_params[:, :, 1] / model_params.sum(axis=2)
        choice_prob = softmax(beta * values, axis=1)

        # Calculate negative log-likelihood
        chosen_prob = choice_prob[np.arange(len(self.choices)), self.choices]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> NDArrayNumber:
        init_beta = np.random.gamma(2, 0.3333)
        return np.array([init_beta])

    def constraints(self):
        A = np.eye(1)
        lb = np.array([0])
        ub = [np.inf]
        return LinearConstraint(A, lb, ub)
