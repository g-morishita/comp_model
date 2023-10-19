import numpy as np
from scipy.optimize import LinearConstraint
from scipy.special import softmax
from typing import Sequence

from .base import MLEstimator, HierarchicalEstimator

from ...type import NDArrayNumber


class QSotfmaxMLE(MLEstimator):
    def __init__(self) -> None:
        """
        This class estimates free parameters of  a simple Q learning model, which makes a choice using softmax function using the maximum likelihood estimator (MLE).
        The free parameters are a learning rate `lr` and an inverse temperature `beta`.
        """
        super().__init__()

    def neg_ll(self, args):
        lr, beta = args

        # Initialize Q-values matrix with zeros
        Q = np.zeros((len(self.choices), self.num_choices))

        # For each trial, calculate delta and update Q-values
        for t in range(1, len(self.choices)):
            a_t = self.choices[t]  # Action taken at time t
            r_t = self.rewards[t]  # Reward received at time t
            delta_t = r_t - Q[t - 1, a_t]

            # Q-value update
            Q[t, a_t] = Q[t - 1, a_t] + lr * delta_t

            # For actions not taken, Q-values remain the same
            for a in range(self.num_choices):
                if a != a_t:
                    Q[t, a] = Q[t - 1, a]

        # Calculate choice probabilities using softmax function
        choice_prob = softmax(beta * Q, axis=1)

        # Calculate negative log-likelihood
        chosen_prob = choice_prob[np.arange(len(self.choices)), self.choices]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        init_lr = np.random.beta(2, 2)
        init_beta = np.random.gamma(2, 0.333)
        return np.array([init_lr, init_beta])

    def constraints(self):
        A = np.eye(2)
        lb = np.array([0, 0])
        ub = [1, np.inf]
        return LinearConstraint(A, lb, ub)


class HierarchicalBayesianQSoftmax(HierarchicalEstimator):
    def __init__(self):
        super().__init__()
        self.stan_file = "stan_files/hierarchical_q_learning.stan"

    def convert_stan_data(
        self,
        num_choices: int,
        choices: NDArrayNumber,
        rewards: NDArrayNumber,
        groups: NDArrayNumber,
    ):
        n_uniq_groups = np.unique(groups)
        n_sessions = choices.shape[0]
        n_trials = choices.shape[1]

        return {
            "N": n_uniq_groups,
            "S": n_sessions,
            "T": n_trials,
            "C": choices.reshape(),
            "R": rewards.reshape(),
        }
