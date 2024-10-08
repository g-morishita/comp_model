import os
from typing import Sequence
from warnings import warn

import numpy as np
from scipy.optimize import LinearConstraint
from scipy.special import softmax

from .base import MLEstimator, HierarchicalEstimator, BayesianEstimator
from ...type import NDArrayNumber


class QSotfmaxMLEWithoutOwnReward(MLEstimator):
    def __init__(self) -> None:
        """
        This class estimates free parameters of  a social Q learning model, which learns from partner's choices and
        rewards and makes a choice using softmax function using the maximum likelihood estimator (MLE).
        The free parameters are a learning rate `lr` and an inverse temperature `beta`.
        """
        super().__init__()

    def neg_ll(self, params: Sequence[int | float]) -> float:
        lr, beta = params

        # Initialize Q-values matrix with zeros
        Q = np.zeros((len(self.your_choices), self.num_choices))
        n_trials = len(self.your_choices)

        # For each trial, calculate delta and update Q-values
        for t in range(1, n_trials):
            current_partner_choice = self.partner_choices[
                t - 1
            ]  # Choice made at time t
            current_partner_reward = self.partner_rewards[
                t - 1
            ]  # Reward received at time t
            delta_t = current_partner_reward - Q[t - 1, current_partner_choice]

            # Q-value update
            Q[t, current_partner_choice] = (
                Q[t - 1, current_partner_choice] + lr * delta_t
            )

            # For actions not taken, Q-values remain the same
            for other_choice in range(self.num_choices):
                if other_choice != current_partner_choice:
                    Q[t, other_choice] = Q[t - 1, other_choice]

        # Calculate choice probabilities using softmax function
        choice_prob = softmax(beta * Q, axis=1)

        # Calculate negative log-likelihood using your own choices not partners!
        chosen_prob = choice_prob[np.arange(1, n_trials), self.your_choices[:-1]]
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


class ForgetfulQSoftmaxMLEWithoutOwnReward(MLEstimator):
    def __init__(self):
        super().__init__()

    def neg_ll(self, params: Sequence[int | float]) -> float:
        lr, beta, forgetfulness = params

        # Initialize Q-values matrix with 1/2
        Q = np.ones((len(self.your_choices), self.num_choices)) / 2
        n_trials = len(self.your_choices)

        # For each trial, calculate delta and update Q-values
        for t in range(1, n_trials):
            current_partner_choice = self.partner_choices[
                t - 1
            ]  # Choice made at time t
            current_partner_reward = self.partner_rewards[
                t - 1
            ]  # Reward received at time t
            delta_t = current_partner_reward - Q[t - 1, current_partner_choice]
            # Q-value update
            Q[t, current_partner_choice] = (
                Q[t - 1, current_partner_choice] + lr * delta_t
            )
            for other_choice in range(self.num_choices):
                if other_choice != current_partner_choice:
                    Q[t, other_choice] = (
                        forgetfulness * Q[0, other_choice]
                        + (1 - forgetfulness) * Q[t - 1, other_choice]
                    )

        # Calculate choice probabilities using softmax function
        choice_prob = softmax(beta * Q, axis=1)

        # Calculate negative log-likelihood using your own choices not partners!
        chosen_prob = choice_prob[np.arange(1, n_trials), self.your_choices[:-1]]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        init_lr = np.random.beta(2, 2)
        init_beta = np.random.gamma(2, 0.333)
        init_forgetfulness = np.random.beta(2, 2)
        return np.array([init_lr, init_beta, init_forgetfulness])

    def constraints(self):
        A = np.eye(3)
        lb = np.array([0, 0, 0])
        ub = [1, np.inf, 1]
        return LinearConstraint(A, lb, ub)


class HierarchicalBayesianForgetfulQSoftmaxMLEWithoutOwnReward(HierarchicalEstimator):
    def __init__(self):
        super().__init__()
        module_path = os.path.dirname(__file__)
        self.stan_file = os.path.join(
            module_path, "stan_files/hierarchical_social_forgetful_q_learning.stan"
        )
        self.group2ind = None

    def convert_stan_data(
        self,
        num_choices: int,
        your_choices: NDArrayNumber,
        your_rewards: NDArrayNumber | None,
        partner_choices: NDArrayNumber,
        partner_rewards: NDArrayNumber | None,
        groups: NDArrayNumber,
    ) -> NDArrayNumber:
        uniq_groups = np.unique(groups)
        n_uniq_groups = uniq_groups.shape[0]
        # Assume every group has the same number of sessions.
        n_sessions_per_group = your_choices.shape[0] // n_uniq_groups
        n_trials = your_choices.shape[1]
        reshaped_your_choices = np.zeros(
            (n_uniq_groups, n_sessions_per_group, n_trials)
        )
        reshaped_partner_choices = np.zeros(
            (n_uniq_groups, n_sessions_per_group, n_trials)
        )
        reshaped_partner_rewards = np.zeros(
            (n_uniq_groups, n_sessions_per_group, n_trials)
        )

        self.group2ind = dict(zip(uniq_groups, np.arange(len(uniq_groups))))

        for g in uniq_groups:
            reshaped_your_choices[self.group2ind[g], :, :] = your_choices[groups == g]
            reshaped_partner_choices[self.group2ind[g], :, :] = partner_choices[
                groups == g
            ]
            reshaped_partner_rewards[self.group2ind[g], :, :] = partner_rewards[
                groups == g
            ]

        stan_data = {
            "N": n_uniq_groups,
            "S": n_sessions_per_group,
            "T": n_trials,
            "NC": num_choices,
            "C": (reshaped_your_choices + 1).astype(int).tolist(),
            "PC": (reshaped_partner_choices + 1).astype(int).tolist(),
            "PR": reshaped_partner_rewards.astype(int).tolist(),
        }
        return stan_data


class QSotfmaxMLEWithOwnRewardSameLr(MLEstimator):
    def __init__(self):
        super().__init__()

    def neg_ll(self, params: Sequence[int | float]) -> float:
        lr, beta = params

        # Initialize Q-values matrix with zeros
        Q = np.ones((len(self.your_choices), self.num_choices)) / 2
        n_trials = len(self.your_choices)

        # For each trial, calculate delta and update Q-values
        for t in range(1, n_trials):
            current_your_choice = self.your_choices[t - 1]  # Choice made at time t
            current_your_reward = self.your_rewards[t - 1]  # Reward received at time t
            delta_t = current_your_reward - Q[t - 1, current_your_choice]

            # Q-value update with your experience
            Q[t, current_your_choice] = Q[t - 1, current_your_choice] + lr * delta_t

            # For actions not taken, Q-values remain the same
            for other_choice in range(self.num_choices):
                if other_choice != current_your_choice:
                    Q[t, other_choice] = Q[t - 1, other_choice]

            current_partner_choice = self.partner_choices[
                t - 1
            ]  # Choice made at time t
            current_partner_reward = self.partner_rewards[
                t - 1
            ]  # Reward received at time t
            delta_t = current_partner_reward - Q[t, current_partner_choice]

            # Q-value update with observed experience
            Q[t, current_partner_choice] = Q[t, current_partner_choice] + lr * delta_t

        # Calculate choice probabilities using softmax function
        choice_prob = softmax(beta * Q, axis=1)

        # Calculate negative log-likelihood using your own choices not partners!
        chosen_prob = choice_prob[np.arange(n_trials), self.your_choices]
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


class QSotfmaxMLEWithOwnReward(MLEstimator):
    def __init__(self):
        super().__init__()

    def neg_ll(self, params: Sequence[int | float]) -> float:
        lr_own, lr_other, beta = params

        # Initialize Q-values matrix with zeros
        Q = np.ones((len(self.your_choices), self.num_choices)) / 2
        n_trials = len(self.your_choices)

        # For each trial, calculate delta and update Q-values
        for t in range(1, n_trials):
            current_your_choice = self.your_choices[t - 1]  # Choice made at time t
            current_your_reward = self.your_rewards[t - 1]  # Reward received at time t
            delta_t = current_your_reward - Q[t - 1, current_your_choice]

            # Q-value update with your experience
            Q[t, current_your_choice] = Q[t - 1, current_your_choice] + lr_own * delta_t

            # For actions not taken, Q-values remain the same
            for other_choice in range(self.num_choices):
                if other_choice != current_your_choice:
                    Q[t, other_choice] = Q[t - 1, other_choice]

            current_partner_choice = self.partner_choices[
                t - 1
            ]  # Choice made at time t
            current_partner_reward = self.partner_rewards[
                t - 1
            ]  # Reward received at time t
            delta_t = current_partner_reward - Q[t, current_partner_choice]

            # Q-value update with observed experience
            Q[t, current_partner_choice] = (
                Q[t, current_partner_choice] + lr_other * delta_t
            )

        # Calculate choice probabilities using softmax function
        choice_prob = softmax(beta * Q, axis=1)

        # Calculate negative log-likelihood using your own choices not partners!
        chosen_prob = choice_prob[np.arange(n_trials), self.your_choices]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        init_lr_own = np.random.beta(2, 2)
        init_lr_other = np.random.beta(2, 2)
        init_beta = np.random.gamma(2, 0.333)
        return np.array([init_lr_own, init_lr_other, init_beta])

    def constraints(self):
        A = np.eye(3)
        lb = np.array([0, 0, 0])
        ub = [1, 1, np.inf]
        return LinearConstraint(A, lb, ub)


class BayesianHierarchicalQSoftmaxWithOwnReward(BayesianEstimator):
    def __init__(self):
        super().__init__()
        module_path = os.path.dirname(__file__)
        self.stan_file = os.path.join(
            module_path, "stan_files/social_q_learning_with_own_rewards.stan"
        )

    def convert_stan_data(
        self,
        num_choices: int,
        your_choices: Sequence[int | float],
        your_rewards: Sequence[int | float] | None,
        partner_choices: Sequence[int | float],
        partner_rewards: Sequence[int | float] | None,
    ) -> dict:
        n_trials = len(your_choices)
        your_choices = np.array(your_choices)
        partner_choices = np.array(partner_choices)
        your_rewards = np.array(your_rewards)
        partner_rewards = np.array(partner_rewards)

        stan_data = {
            "T": n_trials,
            "NC": num_choices,
            "C": (your_choices + 1).astype(int).tolist(),
            "R": your_rewards.astype(int).tolist(),
            "PC": (partner_choices + 1).astype(int).tolist(),
            "PR": partner_rewards.astype(int).tolist(),
        }

        return stan_data


class HierarchicalBayesianQSoftmaxWithoutOwnReward(HierarchicalEstimator):
    def __init__(self):
        super().__init__()
        module_path = os.path.dirname(__file__)
        self.stan_file = os.path.join(
            module_path, "stan_files/hierarchical_social_q_learning.stan"
        )
        self.group2ind = None

    def convert_stan_data(
        self,
        num_choices: int,
        your_choices: NDArrayNumber,
        your_rewards: NDArrayNumber | None,
        partner_choices: NDArrayNumber,
        partner_rewards: NDArrayNumber | None,
        groups: NDArrayNumber,
    ) -> NDArrayNumber:
        uniq_groups = np.unique(groups)
        n_uniq_groups = uniq_groups.shape[0]
        # Assume every group has the same number of sessions.
        n_sessions_per_group = your_choices.shape[0] // n_uniq_groups
        n_trials = your_choices.shape[1]
        reshaped_your_choices = np.zeros(
            (n_uniq_groups, n_sessions_per_group, n_trials)
        )
        reshaped_partner_choices = np.zeros(
            (n_uniq_groups, n_sessions_per_group, n_trials)
        )
        reshaped_partner_rewards = np.zeros(
            (n_uniq_groups, n_sessions_per_group, n_trials)
        )

        self.group2ind = dict(zip(uniq_groups, np.arange(len(uniq_groups))))

        for g in uniq_groups:
            reshaped_your_choices[self.group2ind[g], :, :] = your_choices[groups == g]
            reshaped_partner_choices[self.group2ind[g], :, :] = partner_choices[
                groups == g
            ]
            reshaped_partner_rewards[self.group2ind[g], :, :] = partner_rewards[
                groups == g
            ]

        stan_data = {
            "N": n_uniq_groups,
            "S": n_sessions_per_group,
            "T": n_trials,
            "NC": num_choices,
            "C": (reshaped_your_choices + 1).astype(int).tolist(),
            "PC": (reshaped_partner_choices + 1).astype(int).tolist(),
            "PR": reshaped_partner_rewards.astype(int).tolist(),
        }
        return stan_data


class QSotfmaxInfoBonusMLEWithOwnReward(MLEstimator):
    def __init__(self) -> None:
        """
        This class estimates free parameters of  a social Q learning model, which learns from partner's choices and
        rewards and makes a choice using softmax function using the maximum likelihood estimator (MLE).
        Also, it has an information bonus term.
        The free parameters are
            - a learning rate `lr` and an inverse temperature `beta`.
        """
        super().__init__()

    def neg_ll(self, params: Sequence[int | float]) -> float:
        lr_own, lr_partner, beta, coef_info_bonus = params

        # Initialize Q-values matrix with 1/2
        Q = np.ones((len(self.your_choices), self.num_choices)) / 2
        n_chosen = np.ones((len(self.your_choices), self.num_choices))
        n_trials = len(self.your_choices)

        # For each trial, calculate delta and update Q-values
        for t in range(1, n_trials):
            current_your_choice = self.your_choices[t - 1]
            current_your_reward = self.your_rewards[t - 1]

            delta_t = current_your_reward - Q[t - 1, current_your_choice]
            # Q-value update
            Q[t, current_your_choice] = Q[t - 1, current_your_choice] + lr_own * delta_t

            # increase the number of chosen choices
            n_chosen[t, current_your_choice] = 1 + n_chosen[t - 1, current_your_choice]
            # For actions not taken, n_chosen remain the same
            for other_choice in range(self.num_choices):
                if other_choice != current_your_choice:
                    n_chosen[t, other_choice] = n_chosen[t - 1, other_choice]
                    Q[t, other_choice] = Q[t - 1, other_choice]

            current_partner_choice = self.partner_choices[
                t - 1
            ]  # Choice made at time t
            current_partner_reward = self.partner_rewards[
                t - 1
            ]  # Reward received at time t
            delta_t = current_partner_reward - Q[t, current_partner_choice]
            # Q-value update
            Q[t, current_partner_choice] = (
                Q[t, current_partner_choice] + lr_partner * delta_t
            )
            # increase the number of chosen choices
            n_chosen[t, current_partner_choice] = (
                1 + n_chosen[t, current_partner_choice]
            )

        # Calculate choice probabilities using softmax function
        values = beta * (Q + coef_info_bonus * 1 / np.sqrt(n_chosen))
        choice_prob = softmax(values, axis=1)

        # Calculate negative log-likelihood using your own choices not partners!
        chosen_prob = choice_prob[np.arange(n_trials), self.your_choices]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        init_lr_own = np.random.beta(2, 2)
        init_lr_partner = np.random.beta(2, 2)
        init_beta = np.random.gamma(2, 0.333)
        init_coef_bonus_info = np.random.gamma(2, 0.333)
        return np.array([init_lr_own, init_lr_partner, init_beta, init_coef_bonus_info])

    def constraints(self):
        A = np.eye(4)
        lb = np.array([0, 0, 0, 0])
        ub = [1, 1, np.inf, np.inf]
        return LinearConstraint(A, lb, ub)


class QSotfmaxInfoBonusMLEWithOwnRewardSameLr(MLEstimator):
    def __init__(self) -> None:
        """
        This class estimates free parameters of  a social Q learning model, which learns from partner's choices and
        rewards and makes a choice using softmax function using the maximum likelihood estimator (MLE).
        Also, it has an information bonus term.
        The free parameters are a learning rate `lr` and an inverse temperature `beta`.
        """
        super().__init__()

    def neg_ll(self, params: Sequence[int | float]) -> float:
        lr, beta, coef_info_bonus = params

        # Initialize Q-values matrix with 1/2
        Q = np.ones((len(self.your_choices), self.num_choices)) / 2
        n_chosen = np.ones((len(self.your_choices), self.num_choices))
        n_trials = len(self.your_choices)

        # For each trial, calculate delta and update Q-values
        for t in range(1, n_trials):
            current_your_choice = self.your_choices[t - 1]
            current_your_reward = self.your_rewards[t - 1]

            delta_t = current_your_reward - Q[t - 1, current_your_choice]
            # Q-value update
            Q[t, current_your_choice] = Q[t - 1, current_your_choice] + lr * delta_t

            # increase the number of chosen choices
            n_chosen[t, current_your_choice] = 1 + n_chosen[t - 1, current_your_choice]
            # For actions not taken, n_chosen remain the same
            for other_choice in range(self.num_choices):
                if other_choice != current_your_choice:
                    n_chosen[t, other_choice] = n_chosen[t - 1, other_choice]
                    Q[t, other_choice] = Q[t - 1, other_choice]

            current_partner_choice = self.partner_choices[
                t - 1
            ]  # Choice made at time t
            current_partner_reward = self.partner_rewards[
                t - 1
            ]  # Reward received at time t
            delta_t = current_partner_reward - Q[t, current_partner_choice]
            # Q-value update
            Q[t, current_partner_choice] = Q[t, current_partner_choice] + lr * delta_t
            # increase the number of chosen choices
            n_chosen[t, current_partner_choice] = (
                1 + n_chosen[t, current_partner_choice]
            )

        # Calculate choice probabilities using softmax function
        values = beta * (Q + coef_info_bonus * 1 / np.sqrt(n_chosen))
        choice_prob = softmax(values, axis=1)

        # Calculate negative log-likelihood using your own choices not partners!
        chosen_prob = choice_prob[np.arange(n_trials), self.your_choices]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        init_lr = np.random.beta(2, 2)
        init_beta = np.random.gamma(2, 0.333)
        init_coef_bonus_info = np.random.gamma(2, 0.333)
        return np.array([init_lr, init_beta, init_coef_bonus_info])

    def constraints(self):
        A = np.eye(3)
        lb = np.array([0, 0, 0])
        ub = [1, np.inf, np.inf]
        return LinearConstraint(A, lb, ub)


class ForgetfulQSoftmaxMLEWithOwnReward(MLEstimator):
    def __init__(self):
        super().__init__()

    def neg_ll(self, params: Sequence[int | float]) -> float:
        lr_own, lr_partner, beta, forgetfulness_own, forgetfulness_partner = params

        # Initialize Q-values matrix with 1/2
        Q = np.ones((len(self.your_choices), self.num_choices)) / 2
        n_trials = len(self.your_choices)

        # For each trial, calculate delta and update Q-values
        for t in range(1, n_trials):
            current_your_choice = self.your_choices[t - 1]
            current_your_reward = self.your_rewards[t - 1]

            delta_t = current_your_reward - Q[t - 1, current_your_choice]
            # Q-value update
            Q[t, current_your_choice] = Q[t - 1, current_your_choice] + lr_own * delta_t

            for other_choice in range(self.num_choices):
                if other_choice != current_your_choice:
                    Q[t, other_choice] = (
                        forgetfulness_own * Q[0, other_choice]
                        + (1 - forgetfulness_own) * Q[t - 1, other_choice]
                    )

            current_partner_choice = self.partner_choices[
                t - 1
            ]  # Choice made at time t
            current_partner_reward = self.partner_rewards[
                t - 1
            ]  # Reward received at time t
            delta_t = current_partner_reward - Q[t, current_partner_choice]
            # Q-value update
            Q[t, current_partner_choice] = (
                Q[t, current_partner_choice] + lr_partner * delta_t
            )
            for other_choice in range(self.num_choices):
                if other_choice != current_partner_choice:
                    Q[t, other_choice] = (
                        forgetfulness_partner * Q[0, other_choice]
                        + (1 - forgetfulness_partner) * Q[t, other_choice]
                    )

        # Calculate choice probabilities using softmax function
        choice_prob = softmax(beta * Q, axis=1)

        # Calculate negative log-likelihood using your own choices not partners!
        chosen_prob = choice_prob[np.arange(n_trials), self.your_choices]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        init_lr_own = np.random.beta(2, 2)
        init_lr_partner = np.random.beta(2, 2)
        init_beta = np.random.gamma(2, 0.333)
        init_forgetfulness_own = np.random.beta(2, 2)
        init_forgetfulness_partner = np.random.beta(2, 2)
        return np.array(
            [
                init_lr_own,
                init_lr_partner,
                init_beta,
                init_forgetfulness_own,
                init_forgetfulness_partner,
            ]
        )

    def constraints(self):
        A = np.eye(5)
        lb = np.array([0, 0, 0, 0, 0])
        ub = [1, 1, np.inf, 1, 1]
        return LinearConstraint(A, lb, ub)


class ForgetfulQSoftmaxMLEWithOwnRewardMatchedLrFr(MLEstimator):
    def __init__(self):
        super().__init__()

    def neg_ll(self, params: Sequence[int | float]) -> float:
        lr_own, lr_partner, beta = params
        forgetfulness_own = lr_own
        forgetfulness_partner = lr_partner

        # Initialize Q-values matrix with 1/2
        Q = np.ones((len(self.your_choices), self.num_choices)) / 2
        n_trials = len(self.your_choices)

        # For each trial, calculate delta and update Q-values
        for t in range(1, n_trials):
            current_your_choice = self.your_choices[t - 1]
            current_your_reward = self.your_rewards[t - 1]

            delta_t = current_your_reward - Q[t - 1, current_your_choice]
            # Q-value update
            Q[t, current_your_choice] = Q[t - 1, current_your_choice] + lr_own * delta_t

            for other_choice in range(self.num_choices):
                if other_choice != current_your_choice:
                    Q[t, other_choice] = (
                        forgetfulness_own * Q[0, other_choice]
                        + (1 - forgetfulness_own) * Q[t - 1, other_choice]
                    )

            current_partner_choice = self.partner_choices[
                t - 1
            ]  # Choice made at time t
            current_partner_reward = self.partner_rewards[
                t - 1
            ]  # Reward received at time t
            delta_t = current_partner_reward - Q[t, current_partner_choice]
            # Q-value update
            Q[t, current_partner_choice] = (
                Q[t, current_partner_choice] + lr_partner * delta_t
            )
            for other_choice in range(self.num_choices):
                if other_choice != current_partner_choice:
                    Q[t, other_choice] = (
                        forgetfulness_partner * Q[0, other_choice]
                        + (1 - forgetfulness_partner) * Q[t, other_choice]
                    )

        # Calculate choice probabilities using softmax function
        choice_prob = softmax(beta * Q, axis=1)

        # Calculate negative log-likelihood using your own choices not partners!
        chosen_prob = choice_prob[np.arange(n_trials), self.your_choices]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        init_lr_own = np.random.beta(2, 2)
        init_lr_partner = np.random.beta(2, 2)
        init_beta = np.random.gamma(2, 0.333)
        return np.array([init_lr_own, init_lr_partner, init_beta])

    def constraints(self):
        A = np.eye(3)
        lb = np.array([0, 0, 0])
        ub = [1, 1, np.inf]
        return LinearConstraint(A, lb, ub)


class ForgetfulQSoftmaxMLEWithOwnRewardDiffLrSameFr(MLEstimator):
    def __init__(self):
        super().__init__()

    def neg_ll(self, params: Sequence[int | float]) -> float:
        lr_own, lr_partner, beta, forgetfulness = params

        # Initialize Q-values matrix with 1/2
        Q = np.ones((len(self.your_choices), self.num_choices)) / 2
        n_trials = len(self.your_choices)

        # For each trial, calculate delta and update Q-values
        for t in range(1, n_trials):
            current_your_choice = self.your_choices[t - 1]
            current_your_reward = self.your_rewards[t - 1]

            delta_t = current_your_reward - Q[t - 1, current_your_choice]
            # Q-value update
            Q[t, current_your_choice] = Q[t - 1, current_your_choice] + lr_own * delta_t

            for other_choice in range(self.num_choices):
                if other_choice != current_your_choice:
                    Q[t, other_choice] = (
                        forgetfulness * Q[0, other_choice]
                        + (1 - forgetfulness) * Q[t - 1, other_choice]
                    )

            current_partner_choice = self.partner_choices[
                t - 1
            ]  # Choice made at time t
            current_partner_reward = self.partner_rewards[
                t - 1
            ]  # Reward received at time t
            delta_t = current_partner_reward - Q[t, current_partner_choice]
            # Q-value update
            Q[t, current_partner_choice] = (
                Q[t, current_partner_choice] + lr_partner * delta_t
            )
            for other_choice in range(self.num_choices):
                if other_choice != current_partner_choice:
                    Q[t, other_choice] = (
                        forgetfulness * Q[0, other_choice]
                        + (1 - forgetfulness) * Q[t, other_choice]
                    )

        # Calculate choice probabilities using softmax function
        choice_prob = softmax(beta * Q, axis=1)

        # Calculate negative log-likelihood using your own choices not partners!
        chosen_prob = choice_prob[np.arange(n_trials), self.your_choices]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        init_lr_own = np.random.beta(2, 2)
        init_lr_partner = np.random.beta(2, 2)
        init_forgetfulness = np.random.beta(2, 2)
        init_beta = np.random.gamma(2, 0.333)
        return np.array([init_lr_own, init_lr_partner, init_beta, init_forgetfulness])

    def constraints(self):
        A = np.eye(4)
        lb = np.array([0, 0, 0, 0])
        ub = [1, 1, np.inf, 1]
        return LinearConstraint(A, lb, ub)


class ForgetfulQSoftmaxMLEWithOwnRewardSameLrSameFr(MLEstimator):
    def __init__(self):
        super().__init__()

    def neg_ll(self, params: Sequence[int | float]) -> float:
        lr_own, beta, forgetfulness_own = params
        lr_partner = lr_own
        forgetfulness_partner = forgetfulness_own

        # Initialize Q-values matrix with 1/2
        Q = np.ones((len(self.your_choices), self.num_choices)) / 2
        n_trials = len(self.your_choices)

        # For each trial, calculate delta and update Q-values
        for t in range(1, n_trials):
            current_your_choice = self.your_choices[t - 1]
            current_your_reward = self.your_rewards[t - 1]

            delta_t = current_your_reward - Q[t - 1, current_your_choice]
            # Q-value update
            Q[t, current_your_choice] = Q[t - 1, current_your_choice] + lr_own * delta_t

            for other_choice in range(self.num_choices):
                if other_choice != current_your_choice:
                    Q[t, other_choice] = (
                        forgetfulness_own * Q[0, other_choice]
                        + (1 - forgetfulness_own) * Q[t - 1, other_choice]
                    )

            current_partner_choice = self.partner_choices[
                t - 1
            ]  # Choice made at time t
            current_partner_reward = self.partner_rewards[
                t - 1
            ]  # Reward received at time t
            delta_t = current_partner_reward - Q[t, current_partner_choice]
            # Q-value update
            Q[t, current_partner_choice] = (
                Q[t, current_partner_choice] + lr_partner * delta_t
            )
            for other_choice in range(self.num_choices):
                if other_choice != current_partner_choice:
                    Q[t, other_choice] = (
                        forgetfulness_partner * Q[0, other_choice]
                        + (1 - forgetfulness_partner) * Q[t, other_choice]
                    )

        # Calculate choice probabilities using softmax function
        choice_prob = softmax(beta * Q, axis=1)

        # Calculate negative log-likelihood using your own choices not partners!
        chosen_prob = choice_prob[np.arange(n_trials), self.your_choices]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        init_lr_own = np.random.beta(2, 2)
        init_forgetfulness = np.random.beta(2, 2)
        init_beta = np.random.gamma(2, 0.333)
        return np.array([init_lr_own, init_beta, init_forgetfulness])

    def constraints(self):
        A = np.eye(3)
        lb = np.array([0, 0, 0])
        ub = [1, np.inf, 1]
        return LinearConstraint(A, lb, ub)


class ForgetfulQSoftmaxMLEWithOwnRewardSameLrFr(MLEstimator):
    def __init__(self):
        super().__init__()

    def neg_ll(self, params: Sequence[int | float]) -> float:
        lr, beta = params
        forgetfulness = lr

        # Initialize Q-values matrix with 1/2
        Q = np.ones((len(self.your_choices), self.num_choices)) / 2
        n_trials = len(self.your_choices)

        # For each trial, calculate delta and update Q-values
        for t in range(1, n_trials):
            current_your_choice = self.your_choices[t - 1]
            current_your_reward = self.your_rewards[t - 1]

            delta_t = current_your_reward - Q[t - 1, current_your_choice]
            # Q-value update
            Q[t, current_your_choice] = Q[t - 1, current_your_choice] + lr * delta_t

            for other_choice in range(self.num_choices):
                if other_choice != current_your_choice:
                    Q[t, other_choice] = (
                        forgetfulness * Q[0, other_choice]
                        + (1 - forgetfulness) * Q[t - 1, other_choice]
                    )

            current_partner_choice = self.partner_choices[
                t - 1
            ]  # Choice made at time t
            current_partner_reward = self.partner_rewards[
                t - 1
            ]  # Reward received at time t
            delta_t = current_partner_reward - Q[t, current_partner_choice]
            # Q-value update
            Q[t, current_partner_choice] = Q[t, current_partner_choice] + lr * delta_t
            for other_choice in range(self.num_choices):
                if other_choice != current_partner_choice:
                    Q[t, other_choice] = (
                        forgetfulness * Q[0, other_choice]
                        + (1 - forgetfulness) * Q[t, other_choice]
                    )

        # Calculate choice probabilities using softmax function
        choice_prob = softmax(beta * Q, axis=1)

        # Calculate negative log-likelihood using your own choices not partners!
        chosen_prob = choice_prob[np.arange(n_trials), self.your_choices]
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


class StickyYourChoiceQSotfmaxMLEWithOwnRewardWithSameLr(MLEstimator):
    def __init__(self) -> None:
        super().__init__()

    def neg_ll(self, params: Sequence[int | float]) -> float:
        lr, beta, s = params

        # Initialize Q-values matrix with 1/2
        Q = np.ones((len(self.your_choices), self.num_choices)) / 2
        # Stickiness
        stickiness = np.zeros((len(self.your_choices), self.num_choices))
        stickiness[np.arange(1, len(self.your_choices)), self.your_choices[:-1]] = s
        n_trials = len(self.your_choices)

        # For each trial, calculate delta and update Q-values
        for t in range(1, n_trials):
            current_your_choice = self.your_choices[t - 1]
            current_your_reward = self.your_rewards[t - 1]

            delta_t = current_your_reward - Q[t - 1, current_your_choice]
            # Q-value update
            Q[t, current_your_choice] = Q[t - 1, current_your_choice] + lr * delta_t

            # For actions not taken, n_chosen remain the same
            for other_choice in range(self.num_choices):
                if other_choice != current_your_choice:
                    Q[t, other_choice] = Q[t - 1, other_choice]

            current_partner_choice = self.partner_choices[
                t - 1
            ]  # Choice made at time t
            current_partner_reward = self.partner_rewards[
                t - 1
            ]  # Reward received at time t
            delta_t = current_partner_reward - Q[t, current_partner_choice]
            # Q-value update
            Q[t, current_partner_choice] = Q[t, current_partner_choice] + lr * delta_t

        # Calculate choice probabilities using softmax function
        values = beta * (Q + stickiness)
        choice_prob = softmax(values, axis=1)

        # Calculate negative log-likelihood using your own choices not partners!
        chosen_prob = choice_prob[np.arange(n_trials), self.your_choices]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        init_lr = np.random.beta(2, 2)
        init_beta = np.random.gamma(2, 0.333)
        stickiness = np.random.beta(2, 2)
        return np.array([init_lr, init_beta, stickiness])

    def constraints(self):
        A = np.eye(3)
        lb = np.array([0, 0, 0])
        ub = [1, np.inf, 1]
        return LinearConstraint(A, lb, ub)


class StickyYourChoiceQSotfmaxMLEWithOwnReward(MLEstimator):
    def __init__(self) -> None:
        super().__init__()

    def neg_ll(self, params: Sequence[int | float]) -> float:
        lr_own, lr_partner, beta, s = params

        # Initialize Q-values matrix with 1/2
        Q = np.ones((len(self.your_choices), self.num_choices)) / 2
        # Stickiness
        stickiness = np.zeros((len(self.your_choices), self.num_choices))
        stickiness[np.arange(1, len(self.your_choices)), self.your_choices[:-1]] = s
        n_trials = len(self.your_choices)

        # For each trial, calculate delta and update Q-values
        for t in range(1, n_trials):
            current_your_choice = self.your_choices[t - 1]
            current_your_reward = self.your_rewards[t - 1]

            delta_t = current_your_reward - Q[t - 1, current_your_choice]
            # Q-value update
            Q[t, current_your_choice] = Q[t - 1, current_your_choice] + lr_own * delta_t

            # For actions not taken, n_chosen remain the same
            for other_choice in range(self.num_choices):
                if other_choice != current_your_choice:
                    Q[t, other_choice] = Q[t - 1, other_choice]

            current_partner_choice = self.partner_choices[
                t - 1
            ]  # Choice made at time t
            current_partner_reward = self.partner_rewards[
                t - 1
            ]  # Reward received at time t
            delta_t = current_partner_reward - Q[t, current_partner_choice]
            # Q-value update
            Q[t, current_partner_choice] = (
                Q[t, current_partner_choice] + lr_partner * delta_t
            )

        # Calculate choice probabilities using softmax function
        values = beta * (Q + stickiness)
        choice_prob = softmax(values, axis=1)

        # Calculate negative log-likelihood using your own choices not partners!
        chosen_prob = choice_prob[np.arange(n_trials), self.your_choices]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        init_lr_own = np.random.beta(2, 2)
        init_lr_partner = np.random.beta(2, 2)
        init_beta = np.random.gamma(2, 0.333)
        stickiness = np.random.beta(2, 2)
        return np.array([init_lr_own, init_lr_partner, init_beta, stickiness])

    def constraints(self):
        A = np.eye(4)
        lb = np.array([0, 0, 0, 0])
        ub = [1, 1, np.inf, 1]
        return LinearConstraint(A, lb, ub)


class StickyPartnerChoiceQSotfmaxMLEWithOwnRewardWithSameLr(MLEstimator):
    def __init__(self) -> None:
        super().__init__()

    def neg_ll(self, params: Sequence[int | float]) -> float:
        lr, beta, s = params

        # Initialize Q-values matrix with 1/2
        Q = np.ones((len(self.your_choices), self.num_choices)) / 2
        # Stickiness
        stickiness = np.zeros((len(self.partner_choices), self.num_choices))
        stickiness[np.arange(1, len(self.partner_choices)), self.your_choices[:-1]] = s
        n_trials = len(self.your_choices)

        # For each trial, calculate delta and update Q-values
        for t in range(1, n_trials):
            current_your_choice = self.your_choices[t - 1]
            current_your_reward = self.your_rewards[t - 1]

            delta_t = current_your_reward - Q[t - 1, current_your_choice]
            # Q-value update
            Q[t, current_your_choice] = Q[t - 1, current_your_choice] + lr * delta_t

            # For actions not taken, n_chosen remain the same
            for other_choice in range(self.num_choices):
                if other_choice != current_your_choice:
                    Q[t, other_choice] = Q[t - 1, other_choice]

            current_partner_choice = self.partner_choices[
                t - 1
            ]  # Choice made at time t
            current_partner_reward = self.partner_rewards[
                t - 1
            ]  # Reward received at time t
            delta_t = current_partner_reward - Q[t, current_partner_choice]
            # Q-value update
            Q[t, current_partner_choice] = Q[t, current_partner_choice] + lr * delta_t

        # Calculate choice probabilities using softmax function
        values = beta * (Q + stickiness)
        choice_prob = softmax(values, axis=1)

        # Calculate negative log-likelihood using your own choices not partners!
        chosen_prob = choice_prob[np.arange(n_trials), self.your_choices]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        init_lr = np.random.beta(2, 2)
        init_beta = np.random.gamma(2, 0.333)
        stickiness = np.random.beta(2, 2)
        return np.array([init_lr, init_beta, stickiness])

    def constraints(self):
        A = np.eye(3)
        lb = np.array([0, 0, 0])
        ub = [1, np.inf, 1]
        return LinearConstraint(A, lb, ub)


class StickyPartnerChoiceQSotfmaxMLEWithOwnReward(MLEstimator):
    # TODO: parameter recovery failed
    def __init__(self) -> None:
        super().__init__()

    def neg_ll(self, params: Sequence[int | float]) -> float:
        lr_own, lr_partner, beta, s = params

        # Initialize Q-values matrix with 1/2
        Q = np.ones((len(self.your_choices), self.num_choices)) / 2
        # Stickiness
        stickiness = np.zeros((len(self.partner_choices), self.num_choices))
        stickiness[
            np.arange(1, len(self.partner_choices)), self.partner_choices[:-1]
        ] = s
        n_trials = len(self.your_choices)

        # For each trial, calculate delta and update Q-values
        for t in range(1, n_trials):
            current_your_choice = self.your_choices[t - 1]
            current_your_reward = self.your_rewards[t - 1]

            delta_t = current_your_reward - Q[t - 1, current_your_choice]
            # Q-value update
            Q[t, current_your_choice] = Q[t - 1, current_your_choice] + lr_own * delta_t

            # For actions not taken, n_chosen remain the same
            for other_choice in range(self.num_choices):
                if other_choice != current_your_choice:
                    Q[t, other_choice] = Q[t - 1, other_choice]

            current_partner_choice = self.partner_choices[
                t - 1
            ]  # Choice made at time t
            current_partner_reward = self.partner_rewards[
                t - 1
            ]  # Reward received at time t
            delta_t = current_partner_reward - Q[t, current_partner_choice]
            # Q-value update
            Q[t, current_partner_choice] = (
                Q[t, current_partner_choice] + lr_partner * delta_t
            )

        # Calculate choice probabilities using softmax function
        values = beta * (Q + stickiness)
        choice_prob = softmax(values, axis=1)

        # Calculate negative log-likelihood using your own choices not partners!
        chosen_prob = choice_prob[np.arange(n_trials), self.your_choices]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        init_lr_own = np.random.beta(2, 2)
        init_lr_partner = np.random.beta(2, 2)
        init_beta = np.random.gamma(2, 0.333)
        stickiness = np.random.beta(2, 2)
        return np.array([init_lr_own, init_lr_partner, init_beta, stickiness])

    def constraints(self):
        A = np.eye(4)
        lb = np.array([0, 0, 0, 0])
        ub = [1, 1, np.inf, 1]
        return LinearConstraint(A, lb, ub)


class StickyQSotfmaxMLEWithOwnRewardWithSameLr(MLEstimator):
    # TODO: parameter recovery failed
    def __init__(self) -> None:
        super().__init__()

    def neg_ll(self, params: Sequence[int | float]) -> float:
        lr_own, beta, s_own, s_partner = params
        lr_partner = lr_own

        # Initialize Q-values matrix with 1/2
        Q = np.ones((len(self.your_choices), self.num_choices)) / 2
        # Stickiness for your own choice
        stickiness_own = np.zeros((len(self.your_choices), self.num_choices))
        stickiness_own[
            np.arange(1, len(self.your_choices)), self.partner_choices[:-1]
        ] = s_own
        # Stickiness for partner own choice
        stickiness_partner = np.zeros((len(self.partner_choices), self.num_choices))
        stickiness_partner[
            np.arange(1, len(self.partner_choices)), self.partner_choices[:-1]
        ] = s_partner
        n_trials = len(self.your_choices)

        # For each trial, calculate delta and update Q-values
        for t in range(1, n_trials):
            current_your_choice = self.your_choices[t - 1]
            current_your_reward = self.your_rewards[t - 1]

            delta_t = current_your_reward - Q[t - 1, current_your_choice]
            # Q-value update
            Q[t, current_your_choice] = Q[t - 1, current_your_choice] + lr_own * delta_t

            # For actions not taken, n_chosen remain the same
            for other_choice in range(self.num_choices):
                if other_choice != current_your_choice:
                    Q[t, other_choice] = Q[t - 1, other_choice]

            current_partner_choice = self.partner_choices[
                t - 1
            ]  # Choice made at time t
            current_partner_reward = self.partner_rewards[
                t - 1
            ]  # Reward received at time t
            delta_t = current_partner_reward - Q[t, current_partner_choice]
            # Q-value update
            Q[t, current_partner_choice] = (
                Q[t, current_partner_choice] + lr_partner * delta_t
            )

        # Calculate choice probabilities using softmax function
        values = beta * (Q + stickiness_own + stickiness_partner)
        choice_prob = softmax(values, axis=1)

        # Calculate negative log-likelihood using your own choices not partners!
        chosen_prob = choice_prob[np.arange(n_trials), self.your_choices]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        init_lr_own = np.random.beta(2, 2)
        init_beta = np.random.gamma(2, 0.333)
        stickiness_own = np.random.beta(2, 2)
        stickiness_partner = np.random.beta(2, 2)
        return np.array(
            [
                init_lr_own,
                init_beta,
                stickiness_own,
                stickiness_partner,
            ]
        )

    def constraints(self):
        A = np.eye(4)
        lb = np.array([0, 0, 0, 0])
        ub = [1, np.inf, 1, 1]
        return LinearConstraint(A, lb, ub)


class StickyQSotfmaxMLEWithOwnReward(MLEstimator):
    # TODO: parameter recovery failed
    def __init__(self) -> None:
        super().__init__()

    def neg_ll(self, params: Sequence[int | float]) -> float:
        lr_own, lr_partner, beta, s_own, s_partner = params

        # Initialize Q-values matrix with 1/2
        Q = np.ones((len(self.your_choices), self.num_choices)) / 2
        # Stickiness for your own choice
        stickiness_own = np.zeros((len(self.your_choices), self.num_choices))
        stickiness_own[
            np.arange(1, len(self.your_choices)), self.partner_choices[:-1]
        ] = s_own
        # Stickiness for partner own choice
        stickiness_partner = np.zeros((len(self.partner_choices), self.num_choices))
        stickiness_partner[
            np.arange(1, len(self.partner_choices)), self.partner_choices[:-1]
        ] = s_partner
        n_trials = len(self.your_choices)

        # For each trial, calculate delta and update Q-values
        for t in range(1, n_trials):
            current_your_choice = self.your_choices[t - 1]
            current_your_reward = self.your_rewards[t - 1]

            delta_t = current_your_reward - Q[t - 1, current_your_choice]
            # Q-value update
            Q[t, current_your_choice] = Q[t - 1, current_your_choice] + lr_own * delta_t

            # For actions not taken, n_chosen remain the same
            for other_choice in range(self.num_choices):
                if other_choice != current_your_choice:
                    Q[t, other_choice] = Q[t - 1, other_choice]

            current_partner_choice = self.partner_choices[
                t - 1
            ]  # Choice made at time t
            current_partner_reward = self.partner_rewards[
                t - 1
            ]  # Reward received at time t
            delta_t = current_partner_reward - Q[t, current_partner_choice]
            # Q-value update
            Q[t, current_partner_choice] = (
                Q[t, current_partner_choice] + lr_partner * delta_t
            )

        # Calculate choice probabilities using softmax function
        values = beta * (Q + stickiness_own + stickiness_partner)
        choice_prob = softmax(values, axis=1)

        # Calculate negative log-likelihood using your own choices not partners!
        chosen_prob = choice_prob[np.arange(n_trials), self.your_choices]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        init_lr_own = np.random.beta(2, 2)
        init_lr_partner = np.random.beta(2, 2)
        init_beta = np.random.gamma(2, 0.333)
        stickiness_own = np.random.gamma(2, 0.333)
        stickiness_partner = np.random.gamma(2, 0.333)
        return np.array(
            [
                init_lr_own,
                init_lr_partner,
                init_beta,
                stickiness_own,
                stickiness_partner,
            ]
        )

    def constraints(self):
        A = np.eye(5)
        lb = np.array([0, 0, 0, 0, 0])
        ub = [1, 1, np.inf, 1, 1]
        return LinearConstraint(A, lb, ub)


class StickyQSotfmaxInfoBonusMLEWithOwnRewardSameLrSameStickiness(MLEstimator):
    def __init__(self) -> None:
        """
        This class estimates free parameters of  a social Q learning model, which learns from partner's choices and
        rewards and makes a choice using softmax function using the maximum likelihood estimator (MLE).
        Also, it has an information bonus term.
        The free parameters are a learning rate `lr` and an inverse temperature `beta`.
        """
        super().__init__()

    def neg_ll(self, params: Sequence[int | float]) -> float:
        lr, beta, coef_info_bonus, stickiness = params

        # Initialize Q-values matrix with 1/2
        Q = np.ones((len(self.your_choices), self.num_choices)) / 2
        n_chosen = np.ones((len(self.your_choices), self.num_choices))
        n_trials = len(self.your_choices)

        # For each trial, calculate delta and update Q-values
        for t in range(1, n_trials):
            current_your_choice = self.your_choices[t - 1]
            current_your_reward = self.your_rewards[t - 1]

            delta_t = current_your_reward - Q[t - 1, current_your_choice]
            # Q-value update
            Q[t, current_your_choice] = (
                Q[t - 1, current_your_choice] + lr * delta_t + stickiness
            )

            # increase the number of chosen choices
            n_chosen[t, current_your_choice] = 1 + n_chosen[t - 1, current_your_choice]
            # For actions not taken, n_chosen remain the same
            for other_choice in range(self.num_choices):
                if other_choice != current_your_choice:
                    n_chosen[t, other_choice] = n_chosen[t - 1, other_choice]
                    Q[t, other_choice] = Q[t - 1, other_choice]

            current_partner_choice = self.partner_choices[
                t - 1
            ]  # Choice made at time t
            current_partner_reward = self.partner_rewards[
                t - 1
            ]  # Reward received at time t
            delta_t = current_partner_reward - Q[t, current_partner_choice]
            # Q-value update
            Q[t, current_partner_choice] = (
                Q[t, current_partner_choice] + lr * delta_t + stickiness
            )
            # increase the number of chosen choices
            n_chosen[t, current_partner_choice] = (
                1 + n_chosen[t, current_partner_choice]
            )

        # Calculate choice probabilities using softmax function
        values = beta * (Q + coef_info_bonus * 1 / np.sqrt(n_chosen))
        choice_prob = softmax(values, axis=1)

        # Calculate negative log-likelihood using your own choices not partners!
        chosen_prob = choice_prob[np.arange(n_trials), self.your_choices]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        init_lr = np.random.beta(2, 2)
        init_beta = np.random.gamma(2, 0.333)
        init_coef_bonus_info = np.random.gamma(2, 0.333)
        stickiness = np.random.gamma(2, 0.333)
        return np.array([init_lr, init_beta, init_coef_bonus_info, stickiness])

    def constraints(self):
        A = np.eye(4)
        lb = np.array([0, 0, 0, 0])
        ub = [1, np.inf, np.inf, np.inf]
        return LinearConstraint(A, lb, ub)


class HierarchicalBayesianQSoftmaxWithOwnReward(HierarchicalEstimator):
    def __init__(self):
        super().__init__()
        module_path = os.path.dirname(__file__)
        self.stan_file = os.path.join(
            module_path,
            "stan_files/hierarchical_social_q_learning_with_own_rewards.stan",
        )
        self.group2ind = None

    def convert_stan_data(
        self,
        num_choices: int,
        your_choices: NDArrayNumber,
        your_rewards: NDArrayNumber | None,
        partner_choices: NDArrayNumber,
        partner_rewards: NDArrayNumber | None,
        groups: NDArrayNumber,
    ) -> dict:
        uniq_groups = np.unique(groups)
        n_uniq_groups = uniq_groups.shape[0]
        # Assume every group has the same number of sessions.
        n_sessions_per_group = your_choices.shape[0] // n_uniq_groups
        n_trials = your_choices.shape[1]
        reshaped_your_choices = np.zeros(
            (n_uniq_groups, n_sessions_per_group, n_trials)
        )
        reshaped_your_rewards = np.zeros(
            (n_uniq_groups, n_sessions_per_group, n_trials)
        )
        reshaped_partner_choices = np.zeros(
            (n_uniq_groups, n_sessions_per_group, n_trials)
        )
        reshaped_partner_rewards = np.zeros(
            (n_uniq_groups, n_sessions_per_group, n_trials)
        )

        self.group2ind = dict(zip(uniq_groups, np.arange(len(uniq_groups))))

        for g in uniq_groups:
            reshaped_your_choices[self.group2ind[g], :, :] = your_choices[groups == g]
            reshaped_your_rewards[self.group2ind[g], :, :] = your_rewards[groups == g]
            reshaped_partner_choices[self.group2ind[g], :, :] = partner_choices[
                groups == g
            ]
            reshaped_partner_rewards[self.group2ind[g], :, :] = partner_rewards[
                groups == g
            ]

        stan_data = {
            "N": n_uniq_groups,
            "S": n_sessions_per_group,
            "T": n_trials,
            "NC": num_choices,
            "C": (reshaped_your_choices + 1).astype(int).tolist(),
            "R": reshaped_your_rewards.astype(int).tolist(),
            "PC": (reshaped_partner_choices + 1).astype(int).tolist(),
            "PR": reshaped_partner_rewards.astype(int).tolist(),
        }

        return stan_data


class HierarchicalBayesianForgetfulQSoftmax(HierarchicalBayesianQSoftmaxWithOwnReward):
    def __init__(self, mode=None):
        super().__init__()
        module_path = os.path.dirname(__file__)
        file_name = (
            "stan_files/hierarchical_social_forgetful_q_learning_with_own_rewards.stan"
        )
        if mode == "sameLRsameFR":
            file_name = "stan_files/hierarchical_social_forgetful_q_learning_with_own_rewards_same_lr_same_fr.stan"
        elif mode == "MatchedLRFR":
            file_name = "stan_files/hierarchical_social_forgetful_q_learning_with_own_rewards_matched_lr_fr.stan"
        elif mode == "SameLRFR":
            file_name = "stan_files/hierarchical_social_forgetful_q_learning_with_own_rewards_same_lr_fr.stan"
        self.stan_file = os.path.join(
            module_path,
            file_name,
        )
        self.group2ind = None


class HierarchicalBayesianQSoftmaxWithOwnRewardSameLr(
    HierarchicalBayesianQSoftmaxWithOwnReward
):
    def __init__(
        self,
    ):
        super().__init__()
        module_path = os.path.dirname(__file__)
        self.stan_file = os.path.join(
            module_path,
            "stan_files/hierarchical_social_q_learning_with_own_rewards_same_lr.stan",
        )
        self.group2ind = None


class HierarchicalBayesianQSoftmaxInfoBonusWithOwnRewardSameLr(
    HierarchicalBayesianQSoftmaxWithOwnReward
):
    def __init__(self):
        super().__init__()
        module_path = os.path.dirname(__file__)
        self.stan_file = os.path.join(
            module_path,
            "stan_files/hierarchical_social_q_learning_bonus_term_with_own_rewards_same_lr.stan",
        )
        self.group2ind = None


class HierarchicalBayesianQSoftmaxInfoBonusWithOwnReward(
    HierarchicalBayesianQSoftmaxWithOwnReward
):
    def __init__(self):
        super().__init__()
        module_path = os.path.dirname(__file__)
        self.stan_file = os.path.join(
            module_path,
            "stan_files/hierarchical_social_q_learning_bonus_term_with_own_rewards.stan",
        )
        self.group2ind = None
