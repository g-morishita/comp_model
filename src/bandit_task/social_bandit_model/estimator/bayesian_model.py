import numpy as np
from scipy.optimize import LinearConstraint
from scipy.special import softmax

from .base import MLEstimator, HierarchicalEstimator, BayesianEstimator
from ...type import NDArrayNumber


class BetaModelSoftmaxWithOwnRewardFixedIncrement(MLEstimator):
    def __init__(self):
        super().__init__()

    def neg_ll(self, params):
        beta = params[0]

        n_trials = len(self.your_choices)
        model_params = np.ones((n_trials, self.num_choices, 2))

        for t in range(1, n_trials):
            model_params[t, self.your_choices[t - 1], self.your_rewards[t - 1]] = (
                model_params[t - 1, self.your_choices[t - 1], self.your_rewards[t - 1]]
                + 1
            )
            model_params[
                t, self.your_choices[t - 1], 1 - self.your_rewards[t - 1]
            ] = model_params[
                t - 1, self.your_choices[t - 1], 1 - self.your_rewards[t - 1]
            ]

            # For actions not taken, the parameter values remain the same
            for a in range(self.num_choices):
                if a != self.your_choices[t - 1]:
                    model_params[t, a, :] = model_params[t - 1, a, :]

            model_params[
                t, self.partner_choices[t - 1], self.partner_rewards[t - 1]
            ] += 1

        # Calculate choice probabilities using softmax function
        values = model_params[:, :, 1] / model_params.sum(axis=2)
        choice_prob = softmax(beta * values, axis=1)

        # Calculate negative log-likelihood using your own choices not partners!
        chosen_prob = choice_prob[np.arange(n_trials), self.your_choices]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> NDArrayNumber:
        init_beta = np.random.gamma(2, 0.333)
        return np.array([init_beta])

    def constraints(self):
        A = np.eye(1)
        lb = np.array([0])
        ub = [np.inf]
        return LinearConstraint(A, lb, ub)
