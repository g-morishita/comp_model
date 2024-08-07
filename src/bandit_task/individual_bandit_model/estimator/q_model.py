import os

import numpy as np
from scipy.optimize import LinearConstraint
from scipy.special import softmax

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
            a_t = self.choices[t - 1]  # Action taken at time t
            r_t = self.rewards[t - 1]  # Reward received at time t
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


class ForgetfulQSoftmaxMLE(MLEstimator):
    def __init__(self) -> None:
        """
        This class estimates free parameters of the forgetful Q learning model.
        """
        super().__init__()

    def neg_ll(self, args):
        lr, beta, forgetfulness = args

        # Initialize Q-values matrix with zeros
        Q = np.ones((len(self.choices), self.num_choices)) / 2

        # For each trial, calculate delta and update Q-values
        for t in range(1, len(self.choices)):
            a_t = self.choices[t - 1]  # Action taken at time t
            r_t = self.rewards[t - 1]  # Reward received at time t
            delta_t = r_t - Q[t - 1, a_t]

            # Q-value update with prediction error delta
            Q[t, a_t] = Q[t - 1, a_t] + lr * delta_t

            # For actions not taken, Q-values remain the same
            for a in range(self.num_choices):
                if a != a_t:
                    Q[t, a] = (
                        forgetfulness * Q[0, a] + (1 - forgetfulness) * Q[t - 1, a]
                    )

        # Calculate choice probabilities using softmax function
        choice_prob = softmax(beta * Q, axis=1)

        # Calculate negative log-likelihood
        chosen_prob = choice_prob[np.arange(len(self.choices)), self.choices]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        init_lr = np.random.beta(2, 2)
        init_beta = np.random.gamma(2, 0.333)
        init_forgetfulness = np.random.uniform(2, 2)
        return np.array([init_lr, init_beta, init_forgetfulness])

    def constraints(self):
        A = np.eye(3)
        lb = np.array([0, 0, 0])
        ub = [1, np.inf, 1]
        return LinearConstraint(A, lb, ub)


class StickyQSotfmaxMLE(MLEstimator):
    def __init__(self) -> None:
        """
        This class estimates free parameters of  a sticky Q learning model, which makes a choice using softmax function using the maximum likelihood estimator (MLE).
        The free parameters are a learning rate `lr`, an inverse temperature `beta`, and a stickiness `s`.
        """
        super().__init__()

    def neg_ll(self, args):
        lr, beta, s = args

        # Initialize Q-values matrix with zeros
        Q = np.ones((len(self.choices), self.num_choices)) / 2
        # Stickiness
        stickiness = np.zeros((len(self.choices), self.num_choices))
        stickiness[np.arange(1, len(self.choices)), self.choices[:-1]] = s

        # For each trial, calculate delta and update Q-values
        for t in range(1, len(self.choices)):
            a_t = self.choices[t - 1]  # Action taken at time t
            r_t = self.rewards[t - 1]  # Reward received at time t
            delta_t = r_t - Q[t - 1, a_t]

            # Q-value update with prediction error delta
            Q[t, a_t] = Q[t - 1, a_t] + lr * delta_t

            # For actions not taken, Q-values remain the same
            for a in range(self.num_choices):
                if a != a_t:
                    Q[t, a] = Q[t - 1, a]

        # Calculate choice probabilities using softmax function
        choice_prob = softmax(beta * (Q + stickiness), axis=1)

        # Calculate negative log-likelihood
        chosen_prob = choice_prob[np.arange(len(self.choices)), self.choices]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        init_lr = np.random.beta(2, 2)
        init_beta = np.random.gamma(2, 0.333)
        init_s = np.random.beta(2, 2)
        return np.array([init_lr, init_beta, init_s])

    def constraints(self):
        A = np.eye(3)
        lb = np.array([0, 0, 0])
        ub = [1, np.inf, 1]
        return LinearConstraint(A, lb, ub)


class HierarchicalBayesianQSoftmax(HierarchicalEstimator):
    def __init__(self):
        super().__init__()
        module_path = os.path.dirname(__file__)
        self.stan_file = os.path.join(
            module_path, "stan_files/hierarchical_q_learning.stan"
        )
        self.group2ind = None

    def convert_stan_data(
        self,
        num_choices: int,
        choices: NDArrayNumber,
        rewards: NDArrayNumber,
        groups: NDArrayNumber,
    ):
        uniq_groups = np.unique(groups)
        n_uniq_groups = uniq_groups.shape[0]
        # Assume every group has the same number of sessions.
        n_sessions_per_group = choices.shape[0] // n_uniq_groups
        n_trials = choices.shape[1]
        reshaped_choices = np.zeros((n_uniq_groups, n_sessions_per_group, n_trials))
        reshaped_rewards = np.zeros((n_uniq_groups, n_sessions_per_group, n_trials))

        self.group2ind = dict(zip(uniq_groups, np.arange(len(uniq_groups))))

        for g in uniq_groups:
            reshaped_choices[self.group2ind[g], :, :] = choices[groups == g]
            reshaped_rewards[self.group2ind[g], :, :] = rewards[groups == g]

        stan_data = {
            "N": n_uniq_groups,
            "S": n_sessions_per_group,
            "T": n_trials,
            "NC": np.unique(choices).shape[0],
            "C": (reshaped_choices + 1).astype(int).tolist(),
            "R": reshaped_rewards.astype(int).tolist(),
        }
        return stan_data
