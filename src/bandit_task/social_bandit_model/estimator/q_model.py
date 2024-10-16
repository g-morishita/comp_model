import os
from typing import Sequence
from warnings import warn

import numpy as np
from scipy.optimize import LinearConstraint, Bounds
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

    def session_neg_ll(
        self,
        params: Sequence[int | float],
        your_choices,
        your_rewards,
        partner_choices,
        partner_rewards,
    ) -> float:
        lr_own, lr_other, beta = params

        # Initialize Q-values matrix with zeros
        Q = np.ones((len(your_choices), self.num_choices)) / 2
        n_trials = len(your_choices)

        # For each trial, calculate delta and update Q-values
        for t in range(1, n_trials):
            current_your_choice = your_choices[t - 1]  # Choice made at time t
            current_your_reward = your_rewards[t - 1]  # Reward received at time t
            delta_t = current_your_reward - Q[t - 1, current_your_choice]

            # Q-value update with your experience
            Q[t, current_your_choice] = Q[t - 1, current_your_choice] + lr_own * delta_t

            # For actions not taken, Q-values remain the same
            for other_choice in range(self.num_choices):
                if other_choice != current_your_choice:
                    Q[t, other_choice] = Q[t - 1, other_choice]

            current_partner_choice = partner_choices[t - 1]  # Choice made at time t
            current_partner_reward = partner_rewards[t - 1]  # Reward received at time t
            delta_t = current_partner_reward - Q[t, current_partner_choice]

            # Q-value update with observed experience
            Q[t, current_partner_choice] = (
                Q[t, current_partner_choice] + lr_other * delta_t
            )

        # Calculate choice probabilities using softmax function
        choice_prob = softmax(beta * Q, axis=1)

        # Calculate negative log-likelihood using your own choices not partners!
        chosen_prob = choice_prob[np.arange(n_trials), your_choices]
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

    def session_neg_ll(
        self,
        params: Sequence[int | float],
        your_choices,
        your_rewards,
        partner_choices,
        partner_rewards,
    ) -> float:
        lr_own, lr_partner, beta, coef_info_bonus = params

        # Initialize Q-values matrix with 1/2
        Q = np.ones((len(your_choices), self.num_choices)) / 2
        n_chosen = np.ones((len(your_choices), self.num_choices))
        n_trials = len(your_choices)

        # For each trial, calculate delta and update Q-values
        for t in range(1, n_trials):
            current_your_choice = your_choices[t - 1]
            current_your_reward = your_rewards[t - 1]

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

            current_partner_choice = partner_choices[t - 1]  # Choice made at time t
            current_partner_reward = partner_rewards[t - 1]  # Reward received at time t
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
        chosen_prob = choice_prob[np.arange(n_trials), your_choices]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        init_lr_own = np.random.uniform(0, 1)
        init_lr_partner = np.random.uniform(0, 1)
        init_beta = np.random.gamma(2, 0.333)
        init_coef_bonus_info = np.random.uniform(0, 1)
        return np.array([init_lr_own, init_lr_partner, init_beta, init_coef_bonus_info])

    def constraints(self):
        A = np.eye(4)
        lb = np.array([0, 0, 0, 0])
        ub = [1, 1, np.inf, 1]
        return LinearConstraint(A, lb, ub)


class QSotfmaxInfoBonusMLEWithOwnReward2(MLEstimator):
    def __init__(self) -> None:
        """
        This class estimates free parameters of  a social Q learning model, which learns from partner's choices and
        rewards and makes a choice using softmax function using the maximum likelihood estimator (MLE).
        Also, it has an information bonus term.
        The free parameters are
            - a learning rate `lr` and an inverse temperature `beta`.
        """
        super().__init__()

    def session_neg_ll(
        self,
        params: Sequence[int | float],
        your_choices,
        your_rewards,
        partner_choices,
        partner_rewards,
    ) -> float:
        lr_own, lr_partner, beta, coef_info_bonus = params

        # Initialize Q-values matrix with 1/2
        Q = np.ones((len(your_choices), self.num_choices)) / 2
        n_chosen = np.ones((len(your_choices), self.num_choices))
        n_trials = len(your_choices)

        # For each trial, calculate delta and update Q-values
        for t in range(1, n_trials):
            current_your_choice = your_choices[t - 1]
            current_your_reward = your_rewards[t - 1]

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

            current_partner_choice = partner_choices[t - 1]  # Choice made at time t
            current_partner_reward = partner_rewards[t - 1]  # Reward received at time t
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
        values = beta * (Q + coef_info_bonus * 0.5 / n_chosen)
        choice_prob = softmax(values, axis=1)

        # Calculate negative log-likelihood using your own choices not partners!
        chosen_prob = choice_prob[np.arange(n_trials), your_choices]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        init_lr_own = np.random.uniform(0, 1)
        init_lr_partner = np.random.uniform(0, 1)
        init_beta = np.random.gamma(2, 0.333)
        init_coef_bonus_info = np.random.uniform(0, 1)
        return np.array([init_lr_own, init_lr_partner, init_beta, init_coef_bonus_info])

    def constraints(self):
        A = np.eye(4)
        lb = np.array([0, 0, 0, 0])
        ub = [1, 1, np.inf, 1]
        return LinearConstraint(A, lb, ub)


class QSotfmaxDecayingInfoBonusMLEWithOwnRewardSameDecay(MLEstimator):
    # TODO: parameter recovery failed. only initial_info_bonus and info_decaying_rate
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
        lr_own, lr_partner, beta, initial_info_bonus, info_decaying_rate = params

        n_trials = len(self.your_choices)
        num_actions = self.num_choices

        # Initialize Q-values and info_bonuses
        Q = np.ones((n_trials + 1, num_actions)) * 0.5  # Initial Q-values set to 0.5
        info_bonuses = (
            np.ones((n_trials + 1, num_actions)) * initial_info_bonus
        )  # Initial info bonuses

        neg_log_likelihood = 0.0
        for t in range(n_trials):
            # Calculate action values
            values = Q[t] + info_bonuses[t]
            # Scale by beta
            values_scaled = beta * values
            # Compute choice probabilities using softmax
            choice_probs = softmax(values_scaled)

            # Get the probability of the chosen action
            your_choice = self.your_choices[t]
            choice_prob = choice_probs[your_choice]

            # Accumulate negative log-likelihood
            neg_log_likelihood -= np.log(
                choice_prob + 1e-8
            )  # Add small constant to prevent log(0)

            # Update Q-values for your own choice
            delta_own = self.your_rewards[t] - Q[t, your_choice]
            Q[t + 1] = Q[t]  # Start from previous Q-values
            Q[t + 1, your_choice] += lr_own * delta_own

            # Update Q-values for partner's choice
            partner_choice = self.partner_choices[t]
            delta_partner = self.partner_rewards[t] - Q[t + 1, partner_choice]
            Q[t + 1, partner_choice] += lr_partner * delta_partner

            # Update info bonuses
            info_bonuses[t + 1] = info_bonuses[t]
            # Decay info bonus for your own choice
            info_bonuses[t + 1, your_choice] *= info_decaying_rate
            # Decay info bonus for partner's choice
            info_bonuses[t + 1, partner_choice] *= info_decaying_rate

        return neg_log_likelihood

    def initialize_params(self):
        # Provide initial guesses for the parameters
        init_lr_own = np.random.uniform(0, 1)
        init_lr_partner = np.random.uniform(0, 1)
        init_beta = np.random.uniform(1, 5)
        init_info_bonus = np.random.uniform(0, 1)
        init_decay_rate = np.random.uniform(0, 1)
        return np.array(
            [init_lr_own, init_lr_partner, init_beta, init_info_bonus, init_decay_rate]
        )

    def constraints(self):
        A = np.eye(5)
        lb = np.array([0, 0, 0, -1, 0])
        ub = [1, 1, np.inf, 1, 1]
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

    def session_neg_ll(
        self,
        params: Sequence[int | float],
        your_choices,
        your_rewards,
        partner_choices,
        partner_rewards,
    ) -> float:
        lr_own, lr_partner, beta, forgetfulness_own, forgetfulness_partner = params

        # Initialize Q-values matrix with 1/2
        Q = np.ones((len(your_choices), self.num_choices)) / 2
        n_trials = len(your_choices)

        # For each trial, calculate delta and update Q-values
        for t in range(1, n_trials):
            current_your_choice = your_choices[t - 1]
            current_your_reward = your_rewards[t - 1]

            delta_t = current_your_reward - Q[t - 1, current_your_choice]
            # Q-value update
            Q[t, current_your_choice] = Q[t - 1, current_your_choice] + lr_own * delta_t

            for other_choice in range(self.num_choices):
                if other_choice != current_your_choice:
                    Q[t, other_choice] = (
                        forgetfulness_own * Q[0, other_choice]
                        + (1 - forgetfulness_own) * Q[t - 1, other_choice]
                    )

            current_partner_choice = partner_choices[t - 1]  # Choice made at time t
            current_partner_reward = partner_rewards[t - 1]  # Reward received at time t
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
        chosen_prob = choice_prob[np.arange(n_trials), your_choices]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        init_lr_own = np.random.uniform(0, 1)
        init_lr_partner = np.random.uniform(0, 1)
        init_beta = np.random.gamma(2, 0.333)
        init_forgetfulness_own = np.random.uniform(0, 1)
        init_forgetfulness_partner = np.random.uniform(0, 1)
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
    def __init__(self) -> None:
        super().__init__()

    def session_neg_ll(
        self,
        params: Sequence[int | float],
        your_choices,
        your_rewards,
        partner_choices,
        partner_rewards,
    ) -> float:
        lr_own, lr_partner, beta, s = params

        # Initialize Q-values matrix with 1/2
        Q = np.ones((len(your_choices), self.num_choices)) / 2
        # Stickiness
        stickiness = np.zeros((len(partner_choices), self.num_choices))
        stickiness[np.arange(1, len(partner_choices)), partner_choices[:-1]] = s
        n_trials = len(your_choices)

        # For each trial, calculate delta and update Q-values
        for t in range(1, n_trials):
            current_your_choice = your_choices[t - 1]
            current_your_reward = your_rewards[t - 1]

            delta_t = current_your_reward - Q[t - 1, current_your_choice]
            # Q-value update
            Q[t, current_your_choice] = Q[t - 1, current_your_choice] + lr_own * delta_t

            # For actions not taken, n_chosen remain the same
            for other_choice in range(self.num_choices):
                if other_choice != current_your_choice:
                    Q[t, other_choice] = Q[t - 1, other_choice]

            current_partner_choice = partner_choices[t - 1]  # Choice made at time t
            current_partner_reward = partner_rewards[t - 1]  # Reward received at time t
            delta_t = current_partner_reward - Q[t, current_partner_choice]
            # Q-value update
            Q[t, current_partner_choice] = (
                Q[t, current_partner_choice] + lr_partner * delta_t
            )

        # Calculate choice probabilities using softmax function
        values = beta * (Q + stickiness)
        choice_prob = softmax(values, axis=1)

        # Calculate negative log-likelihood using your own choices not partners!
        chosen_prob = choice_prob[np.arange(n_trials), your_choices]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        init_lr_own = np.random.uniform(0, 1)
        init_lr_partner = np.random.uniform(0, 1)
        init_beta = np.random.gamma(2, 0.333)
        stickiness = np.random.uniform(0, 1)
        return np.array([init_lr_own, init_lr_partner, init_beta, stickiness])

    def constraints(self):
        A = np.eye(4)
        lb = np.array([0, 0, 0, 0])
        ub = [1, 1, np.inf, 1]
        return LinearConstraint(A, lb, ub)


class StickyQSotfmaxMLEWithOwnRewardWithSameLr(MLEstimator):
    # TODO: parameter recovery failed
    # TODO: I haven't confimred it yet but I think it has been fixed.
    def __init__(self) -> None:
        super().__init__()

    def session_neg_ll(self, params: Sequence[int | float]) -> float:
        lr_own, beta, s_own, s_partner = params
        lr_partner = lr_own

        # Initialize Q-values matrix with 1/2
        Q = np.ones((len(self.your_choices), self.num_choices)) / 2

        # Stickiness for your own choice
        stickiness_own = np.zeros((len(self.your_choices), self.num_choices))
        stickiness_own[
            np.arange(1, len(self.your_choices)), self.your_choices[:-1]
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
    # TODO: Check parameter recovery
    def __init__(self) -> None:
        super().__init__()

    def session_neg_ll(
        self,
        params: Sequence[int | float],
        your_choices,
        your_rewards,
        partner_choices,
        partner_rewards,
    ) -> float:
        lr_own, lr_partner, beta, s_own, s_partner = params

        # Initialize Q-values matrix with 1/2
        Q = np.ones((len(your_choices), self.num_choices)) / 2

        # Stickiness for your own choice
        stickiness_own = np.zeros((len(your_choices), self.num_choices))
        stickiness_own[np.arange(1, len(your_choices)), your_choices[:-1]] = s_own

        # Stickiness for partner own choice
        stickiness_partner = np.zeros((len(partner_choices), self.num_choices))
        stickiness_partner[
            np.arange(1, len(partner_choices)), partner_choices[:-1]
        ] = s_partner
        n_trials = len(your_choices)

        # For each trial, calculate delta and update Q-values
        for t in range(1, n_trials):
            current_your_choice = your_choices[t - 1]
            current_your_reward = your_rewards[t - 1]

            delta_t = current_your_reward - Q[t - 1, current_your_choice]
            # Q-value update
            Q[t, current_your_choice] = Q[t - 1, current_your_choice] + lr_own * delta_t

            # For actions not taken, n_chosen remain the same
            for other_choice in range(self.num_choices):
                if other_choice != current_your_choice:
                    Q[t, other_choice] = Q[t - 1, other_choice]

            current_partner_choice = partner_choices[t - 1]  # Choice made at time t
            current_partner_reward = partner_rewards[t - 1]  # Reward received at time t
            delta_t = current_partner_reward - Q[t, current_partner_choice]
            # Q-value update
            Q[t, current_partner_choice] = (
                Q[t, current_partner_choice] + lr_partner * delta_t
            )

        # Calculate choice probabilities using softmax function
        values = beta * (Q + stickiness_own + stickiness_partner)
        choice_prob = softmax(values, axis=1)

        # Calculate negative log-likelihood using your own choices not partners!
        chosen_prob = choice_prob[np.arange(n_trials), your_choices]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        init_lr_own = np.random.uniform(0, 1)
        init_lr_partner = np.random.uniform(0, 1)
        init_beta = np.random.gamma(2, 0.333)
        stickiness_own = np.random.uniform(0, 1)
        stickiness_partner = np.random.uniform(0, 1)
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


class StickyQSotfmaxMLEWithOwnRewardSameS(MLEstimator):
    # TODO: parameter recovery failed
    # TODO: I haven't confimred it yet but I think it has been fixed.
    def __init__(self) -> None:
        super().__init__()

    def session_neg_ll(
        self,
        params: Sequence[int | float],
        your_choices,
        your_rewards,
        partner_choices,
        partner_rewards,
    ) -> float:
        lr_own, lr_partner, beta, s = params

        # Initialize Q-values matrix with 1/2
        Q = np.ones((len(your_choices), self.num_choices)) / 2

        # Stickiness for your own choice
        stickiness_own = np.zeros((len(your_choices), self.num_choices))
        stickiness_own[np.arange(1, len(your_choices)), your_choices[:-1]] = s

        # Stickiness for partner own choice
        stickiness_partner = np.zeros((len(partner_choices), self.num_choices))
        stickiness_partner[np.arange(1, len(partner_choices)), partner_choices[:-1]] = s
        n_trials = len(your_choices)

        # For each trial, calculate delta and update Q-values
        for t in range(1, n_trials):
            current_your_choice = your_choices[t - 1]
            current_your_reward = your_rewards[t - 1]

            delta_t = current_your_reward - Q[t - 1, current_your_choice]
            # Q-value update
            Q[t, current_your_choice] = Q[t - 1, current_your_choice] + lr_own * delta_t

            # For actions not taken, n_chosen remain the same
            for other_choice in range(self.num_choices):
                if other_choice != current_your_choice:
                    Q[t, other_choice] = Q[t - 1, other_choice]

            current_partner_choice = partner_choices[t - 1]  # Choice made at time t
            current_partner_reward = partner_rewards[t - 1]  # Reward received at time t
            delta_t = current_partner_reward - Q[t, current_partner_choice]
            # Q-value update
            Q[t, current_partner_choice] = (
                Q[t, current_partner_choice] + lr_partner * delta_t
            )

        # Calculate choice probabilities using softmax function
        values = beta * (Q + stickiness_own + stickiness_partner)
        choice_prob = softmax(values, axis=1)

        # Calculate negative log-likelihood using your own choices not partners!
        chosen_prob = choice_prob[np.arange(n_trials), your_choices]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        init_lr_own = np.random.beta(2, 2)
        init_lr_partner = np.random.beta(2, 2)
        init_beta = np.random.gamma(2, 0.333)
        s = np.random.gamma(2, 0.333)
        return np.array([init_lr_own, init_lr_partner, init_beta, s])

    def constraints(self):
        A = np.eye(4)
        lb = np.array([0, 0, 0, -np.inf])
        ub = [1, 1, np.inf, np.inf]
        return LinearConstraint(A, lb, ub)


class StickyYourChoiceQSotfmaxInfoBonusMLEWithOwnReward(MLEstimator):
    # TODO: Information bonus term and stickiness term might have similar effects
    #  so they might cause identifiability problem.
    def __init__(self) -> None:
        """
        This class estimates free parameters of  a social Q learning model, which learns from partner's choices and
        rewards and makes a choice using softmax function using the maximum likelihood estimator (MLE).
        Also, it has an information bonus term.
        The free parameters are a learning rate `lr` and an inverse temperature `beta`.
        """
        super().__init__()

    def session_neg_ll(
        self,
        params: Sequence[int | float],
        your_choices,
        your_rewards,
        partner_choices,
        partner_rewards,
    ) -> float:
        lr_own, lr_partner, beta, coef_info_bonus, stickiness_own = params

        n_trials = len(your_choices)
        num_choices = self.num_choices

        # Initialize Q-values and n_chosen
        Q = np.ones((n_trials + 1, num_choices)) / 2
        n_chosen = np.ones(num_choices)

        # Initialize variables to store previous choices
        previous_own_choice = None

        choice_prob = np.zeros((n_trials, num_choices))

        for t in range(n_trials):
            # Compute the information bonus
            info_bonus = coef_info_bonus * (1 / np.sqrt(n_chosen))

            # Calculate action values for softmax
            values = Q[t] + info_bonus

            # Add stickiness to the appropriate action values
            if previous_own_choice is not None:
                values[previous_own_choice] += stickiness_own

            # Scale by beta
            values *= beta

            # Compute choice probabilities
            probs = softmax(values)
            choice_prob[t] = probs

            # Update n_chosen
            n_chosen[your_choices[t]] += 1
            n_chosen[partner_choices[t]] += 1

            # Update Q-values
            delta_own = your_rewards[t] - Q[t, your_choices[t]]
            Q[t + 1] = Q[t]
            Q[t + 1, your_choices[t]] += lr_own * delta_own

            delta_partner = partner_rewards[t] - Q[t + 1, partner_choices[t]]
            Q[t + 1, partner_choices[t]] += lr_partner * delta_partner

            # Update previous choices
            previous_own_choice = your_choices[t]

        # Calculate negative log-likelihood
        chosen_prob = choice_prob[np.arange(n_trials), your_choices]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        init_lr_own = np.random.beta(2, 2)
        init_lr_partner = np.random.beta(2, 2)
        init_beta = np.random.gamma(2, 0.333)
        init_coef_bonus_info = np.random.gamma(2, 0.333)
        stickiness_own = np.random.gamma(2, 0.333)
        return np.array(
            [
                init_lr_own,
                init_lr_partner,
                init_beta,
                init_coef_bonus_info,
                stickiness_own,
            ]
        )

    def constraints(self):
        A = np.eye(5)
        lb = np.array([0, 0, 0, -np.inf, -np.inf])
        ub = [1, 1, np.inf, np.inf, np.inf]
        return LinearConstraint(A, lb, ub)


class StickyPartnerChoiceQSotfmaxInfoBonusMLEWithOwnReward(MLEstimator):
    # TODO: Information bonus term and stickiness term might have similar effects
    #  so they might cause identifiability problem.
    def __init__(self) -> None:
        """
        This class estimates free parameters of  a social Q learning model, which learns from partner's choices and
        rewards and makes a choice using softmax function using the maximum likelihood estimator (MLE).
        Also, it has an information bonus term.
        The free parameters are a learning rate `lr` and an inverse temperature `beta`.
        """
        super().__init__()

    def session_neg_ll(
        self,
        params: Sequence[int | float],
        your_choices,
        your_rewards,
        partner_choices,
        partner_rewards,
    ) -> float:
        lr_own, lr_partner, beta, coef_info_bonus, stickiness_partner = params

        n_trials = len(your_choices)
        num_choices = self.num_choices

        # Initialize Q-values and n_chosen
        Q = np.ones((n_trials + 1, num_choices)) / 2
        n_chosen = np.ones(num_choices)

        # Initialize variables to store previous choices
        previous_partner_choice = None

        choice_prob = np.zeros((n_trials, num_choices))

        for t in range(n_trials):
            # Compute the information bonus
            info_bonus = coef_info_bonus * (1 / np.sqrt(n_chosen))

            # Calculate action values for softmax
            values = Q[t] + info_bonus

            # Add stickiness to the appropriate action values
            if previous_partner_choice is not None:
                values[previous_partner_choice] += stickiness_partner

            # Scale by beta
            values *= beta

            # Compute choice probabilities
            probs = softmax(values)
            choice_prob[t] = probs

            # Update n_chosen
            n_chosen[your_choices[t]] += 1
            n_chosen[partner_choices[t]] += 1

            # Update Q-values
            delta_own = your_rewards[t] - Q[t, your_choices[t]]
            Q[t + 1] = Q[t]
            Q[t + 1, your_choices[t]] += lr_own * delta_own

            delta_partner = partner_rewards[t] - Q[t + 1, partner_choices[t]]
            Q[t + 1, partner_choices[t]] += lr_partner * delta_partner

            # Update previous choices
            previous_own_choice = your_choices[t]

        # Calculate negative log-likelihood
        chosen_prob = choice_prob[np.arange(n_trials), your_choices]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        init_lr_own = np.random.beta(2, 2)
        init_lr_partner = np.random.beta(2, 2)
        init_beta = np.random.gamma(2, 0.333)
        init_coef_bonus_info = np.random.beta(2, 2)
        stickiness_partner = np.random.beta(2, 2)
        return np.array(
            [
                init_lr_own,
                init_lr_partner,
                init_beta,
                init_coef_bonus_info,
                stickiness_partner,
            ]
        )

    def constraints(self):
        A = np.eye(5)
        lb = np.array([0, 0, 0.001, 0, 0])
        ub = [1, 1, 20, 1, 1]
        return LinearConstraint(A, lb, ub)


class StickyQSoftmaxInfoBonusMLEWithOwnReward(MLEstimator):
    def __init__(self) -> None:
        super().__init__()

    def session_neg_ll(
        self,
        params: Sequence[int | float],
        your_choices,
        your_rewards,
        partner_choices,
        partner_rewards,
    ) -> float:
        (
            lr_own,
            lr_partner,
            beta,
            coef_info_bonus,
            stickiness_own,
            stickiness_partner,
        ) = params

        n_trials = len(your_choices)
        num_choices = self.num_choices

        # Initialize Q-values and n_chosen
        Q = np.ones((n_trials + 1, num_choices)) / 2
        n_chosen = np.ones(num_choices)

        # Initialize variables to store previous choices
        previous_own_choice = None
        previous_partner_choice = None

        choice_prob = np.zeros((n_trials, num_choices))

        for t in range(n_trials):
            # Compute the information bonus
            info_bonus = coef_info_bonus * (1 / np.sqrt(n_chosen))

            # Calculate action values for softmax
            values = Q[t] + info_bonus

            # Add stickiness to the appropriate action values
            if previous_own_choice is not None:
                values[previous_own_choice] += stickiness_own
            if previous_partner_choice is not None:
                values[previous_partner_choice] += stickiness_partner

            # Scale by beta
            values *= beta

            # Compute choice probabilities
            probs = softmax(values)
            choice_prob[t] = probs

            # Update n_chosen
            n_chosen[your_choices[t]] += 1
            n_chosen[partner_choices[t]] += 1

            # Update Q-values
            delta_own = your_rewards[t] - Q[t, your_choices[t]]
            Q[t + 1] = Q[t]
            Q[t + 1, your_choices[t]] += lr_own * delta_own

            delta_partner = partner_rewards[t] - Q[t + 1, partner_choices[t]]
            Q[t + 1, partner_choices[t]] += lr_partner * delta_partner

            # Update previous choices
            previous_own_choice = your_choices[t]
            previous_partner_choice = partner_choices[t]

        # Calculate negative log-likelihood
        chosen_prob = choice_prob[np.arange(n_trials), your_choices]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        init_lr_own = np.random.uniform(0, 1)
        init_lr_partner = np.random.uniform(0, 1)
        init_beta = np.random.gamma(2, 0.333)
        init_coef_bonus_info = np.random.uniform(0, 1)
        init_stickiness_own = np.random.uniform(0, 1)
        init_stickiness_partner = np.random.uniform(0, 1)
        return np.array(
            [
                init_lr_own,
                init_lr_partner,
                init_beta,
                init_coef_bonus_info,
                init_stickiness_own,
                init_stickiness_partner,
            ]
        )

    def constraints(self):
        A = np.eye(6)
        lb = np.array([0, 0, 0, 0, 0, 0])
        ub = [1, 1, np.inf, 1, 1, 1]
        return LinearConstraint(A, lb, ub)


class StickyForgetfulQSoftmaxMLEWithOwnReward(MLEstimator):
    def __init__(self):
        super().__init__()

    def session_neg_ll(
        self,
        params: Sequence[float],
        your_choices,
        your_rewards,
        partner_choices,
        partner_rewards,
    ) -> float:
        """
        Calculate the negative log-likelihood for a single session.

        Parameters:
            params: Sequence[float]
                A sequence of parameters to estimate:
                [lr_own, lr_partner, beta, forgetful_own, forgetful_partner,
                 stickiness_own, stickiness_partner]
            your_choices: np.ndarray
                Array of your choices.
            your_rewards: np.ndarray
                Array of your rewards.
            partner_choices: np.ndarray
                Array of your partner's choices.
            partner_rewards: np.ndarray
                Array of your partner's rewards.

        Returns:
            nll: float
                The negative log-likelihood for the session.
        """
        # Unpack parameters
        (
            lr_own,
            lr_partner,
            beta,
            forgetful_own,
            forgetful_partner,
            stickiness_own,
            stickiness_partner,
        ) = params

        n_trials = len(your_choices)
        num_choices = self.num_choices

        # Initialize Q-values
        initial_value = 0.5
        Q = np.ones((n_trials + 1, num_choices)) * initial_value
        initial_values = np.ones(num_choices) * initial_value

        # Initialize variables to store previous choices
        previous_own_choice = None
        previous_partner_choice = None

        # Initialize array to store choice probabilities
        choice_prob = np.zeros((n_trials, num_choices))

        for t in range(n_trials):
            # Get current values
            values = Q[t].copy()

            # Add stickiness effects
            if previous_own_choice is not None:
                values[previous_own_choice] += stickiness_own
            if previous_partner_choice is not None:
                values[previous_partner_choice] += stickiness_partner

            # Scale by beta and compute choice probabilities
            values_scaled = beta * values
            probs = softmax(values_scaled)
            choice_prob[t] = probs

            # Update previous choices
            previous_own_choice = your_choices[t]
            previous_partner_choice = partner_choices[t]

            # Update Q-values for own choice
            Q[t + 1] = Q[t]
            delta_own = your_rewards[t] - Q[t, your_choices[t]]
            Q[t + 1, your_choices[t]] += lr_own * delta_own

            # Apply forgetting to other actions after own choice
            for action in range(num_choices):
                if action != your_choices[t]:
                    Q[t + 1, action] = (
                        forgetful_own * initial_values[action]
                        + (1 - forgetful_own) * Q[t + 1, action]
                    )

            # Update Q-values for partner's choice
            delta_partner = partner_rewards[t] - Q[t + 1, partner_choices[t]]
            Q[t + 1, partner_choices[t]] += lr_partner * delta_partner

            # Apply forgetting to other actions after partner's choice
            for action in range(num_choices):
                if action != partner_choices[t]:
                    Q[t + 1, action] = (
                        forgetful_partner * initial_values[action]
                        + (1 - forgetful_partner) * Q[t + 1, action]
                    )

        # Calculate negative log-likelihood
        chosen_prob = choice_prob[np.arange(n_trials), your_choices]
        nll = -np.sum(np.log(chosen_prob + 1e-8))

        return nll

    def initialize_params(self) -> np.ndarray:
        """
        Provide reasonable initial guesses for the parameters.

        Returns:
            init_params: np.ndarray
                Array of initial parameter guesses.
        """
        init_lr_own = np.random.uniform(0, 1)
        init_lr_partner = np.random.uniform(0, 1)
        init_beta = np.random.uniform(1e-3, 5)
        init_forgetful_own = np.random.uniform(0, 1)
        init_forgetful_partner = np.random.uniform(0, 1)
        init_stickiness_own = np.random.uniform(-1, 1)
        init_stickiness_partner = np.random.uniform(-1, 1)

        return np.array(
            [
                init_lr_own,
                init_lr_partner,
                init_beta,
                init_forgetful_own,
                init_forgetful_partner,
                init_stickiness_own,
                init_stickiness_partner,
            ]
        )

    def constraints(self):
        """
        Define constraints for the optimization problem.

        Returns:
            constraints: LinearConstraint
                Constraints for the optimizer.
        """
        n_params = 7  # Number of parameters to estimate
        A = np.eye(n_params)
        lb = [0, 0, 1e-3, 0, 0, 0, 0]  # Lower bounds
        ub = [1, 1, np.inf, 1, 1, 1, 1]  # Upper bounds
        return LinearConstraint(A, lb, ub)


class ForgetfulQSoftmaxBonusMLEWithOwnReward(MLEstimator):
    def __init__(self) -> None:
        super().__init__()

    def session_neg_ll(
        self,
        params: Sequence[int | float],
        your_choices,
        your_rewards,
        partner_choices,
        partner_rewards,
    ) -> float:
        lr_own, lr_partner, beta, coef_info_bonus, f_own, f_partner = params

        n_trials = len(your_choices)
        num_choices = self.num_choices

        # Initialize Q-values and initial values
        initial_value = 0.5
        Q = np.ones((n_trials + 1, num_choices)) * initial_value
        initial_values = np.ones(num_choices) * initial_value
        n_chosen = np.ones(num_choices)

        # Initialize array to store choice probabilities
        choice_prob = np.zeros((n_trials, num_choices))

        for t in range(n_trials):
            # Compute the information bonus
            info_bonus = coef_info_bonus * (1 / np.sqrt(n_chosen))

            # Calculate action values for softmax
            values = Q[t] + info_bonus

            # Scale by beta and compute choice probabilities
            probs = softmax(beta * values)
            choice_prob[t] = probs

            # Update n_chosen
            n_chosen[your_choices[t]] += 1
            n_chosen[partner_choices[t]] += 1

            # Copy previous Q-values for update
            Q[t + 1] = Q[t]

            # Learn from own choice
            delta_own = your_rewards[t] - Q[t, your_choices[t]]
            Q[t + 1, your_choices[t]] += lr_own * delta_own

            # Apply forgetting to other actions after own choice
            for action in range(num_choices):
                if action != your_choices[t]:
                    Q[t + 1, action] = (
                        f_own * initial_values[action] + (1 - f_own) * Q[t + 1, action]
                    )

            # Learn from partner's choice
            delta_partner = partner_rewards[t] - Q[t + 1, partner_choices[t]]
            Q[t + 1, partner_choices[t]] += lr_partner * delta_partner

            # Apply forgetting to other actions after partner's choice
            for action in range(num_choices):
                if action != partner_choices[t]:
                    Q[t + 1, action] = (
                        f_partner * initial_values[action]
                        + (1 - f_partner) * Q[t + 1, action]
                    )

        # Calculate negative log-likelihood
        chosen_prob = choice_prob[np.arange(n_trials), your_choices]
        nll = -np.sum(np.log(chosen_prob + 1e-8))

        return nll

    def initialize_params(self) -> np.ndarray:
        # Provide reasonable initial parameter guesses
        init_lr_own = np.random.uniform(0, 1)
        init_lr_partner = np.random.uniform(0, 1)
        init_beta = np.random.uniform(1e-3, 5)
        init_coef_info_bonus = np.random.uniform(0, 1)
        init_f_own = np.random.uniform(0, 1)
        init_f_partner = np.random.uniform(0, 1)
        return np.array(
            [
                init_lr_own,
                init_lr_partner,
                init_beta,
                init_coef_info_bonus,
                init_f_own,
                init_f_partner,
            ]
        )

    def constraints(self):
        # Set parameter bounds
        A = np.eye(6)
        lb = [0, 0, 0, 0, 0, 0]
        ub = [1, 1, np.inf, 1, 1, 1]
        return LinearConstraint(A, lb, ub)


class ForgetfulStickyQSoftmaxBonusMLEWithOwnReward(MLEstimator):
    def __init__(self) -> None:
        super().__init__()

    def session_neg_ll(
        self,
        params: Sequence[float],
        your_choices,
        your_rewards,
        partner_choices,
        partner_rewards,
    ) -> float:
        """
        Calculate the negative log-likelihood for a single session.

        Parameters:
            params: Sequence[float]
                A sequence of parameters to estimate:
                [lr_own, lr_partner, beta, coef_info_bonus, forgetful_own,
                 forgetful_partner, stickiness_own, stickiness_partner]
            your_choices: np.ndarray
                Array of your choices.
            your_rewards: np.ndarray
                Array of your rewards.
            partner_choices: np.ndarray
                Array of your partner's choices.
            partner_rewards: np.ndarray
                Array of your partner's rewards.

        Returns:
            nll: float
                The negative log-likelihood for the session.
        """
        # Unpack parameters
        (
            lr_own,
            lr_partner,
            beta,
            coef_info_bonus,
            forgetful_own,
            forgetful_partner,
            stickiness_own,
            stickiness_partner,
        ) = params

        n_trials = len(your_choices)
        num_choices = self.num_choices

        # Initialize Q-values and other variables
        initial_value = 0.5
        Q = np.ones((n_trials + 1, num_choices)) * initial_value
        initial_values = np.ones(num_choices) * initial_value
        n_chosen = np.ones(num_choices)

        # Initialize previous choices for stickiness
        previous_own_choice = None
        previous_partner_choice = None

        # Initialize array to store choice probabilities
        choice_prob = np.zeros((n_trials, num_choices))

        for t in range(n_trials):
            # Compute the information bonus
            info_bonus = coef_info_bonus / np.sqrt(n_chosen)

            # Calculate action values
            values = Q[t] + info_bonus

            # Add stickiness effects
            if previous_own_choice is not None:
                values[previous_own_choice] += stickiness_own
            if previous_partner_choice is not None:
                values[previous_partner_choice] += stickiness_partner

            # Scale by beta and compute choice probabilities
            probs = softmax(beta * values)
            choice_prob[t] = probs

            # Update n_chosen
            n_chosen[your_choices[t]] += 1
            n_chosen[partner_choices[t]] += 1

            # Copy previous Q-values for update
            Q[t + 1] = Q[t]

            # Learn from own choice
            delta_own = your_rewards[t] - Q[t, your_choices[t]]
            Q[t + 1, your_choices[t]] += lr_own * delta_own

            # Apply forgetting to other actions after own choice
            for action in range(num_choices):
                if action != your_choices[t]:
                    Q[t + 1, action] = (
                        forgetful_own * initial_values[action]
                        + (1 - forgetful_own) * Q[t + 1, action]
                    )

            # Update previous own choice
            previous_own_choice = your_choices[t]

            # Learn from partner's choice
            delta_partner = partner_rewards[t] - Q[t + 1, partner_choices[t]]
            Q[t + 1, partner_choices[t]] += lr_partner * delta_partner

            # Apply forgetting to other actions after partner's choice
            for action in range(num_choices):
                if action != partner_choices[t]:
                    Q[t + 1, action] = (
                        forgetful_partner * initial_values[action]
                        + (1 - forgetful_partner) * Q[t + 1, action]
                    )

            # Update previous partner choice
            previous_partner_choice = partner_choices[t]

        # Calculate negative log-likelihood
        chosen_prob = choice_prob[np.arange(n_trials), your_choices]
        nll = -np.sum(np.log(chosen_prob + 1e-8))

        return nll

    def initialize_params(self) -> np.ndarray:
        """
        Provide reasonable initial guesses for the parameters.

        Returns:
            init_params: np.ndarray
                Array of initial parameter guesses.
        """
        init_lr_own = np.random.uniform(0, 1)
        init_lr_partner = np.random.uniform(0, 1)
        init_beta = np.random.uniform(1, 5)
        init_coef_info_bonus = np.random.uniform(0, 1)
        init_forgetful_own = np.random.uniform(0, 1)
        init_forgetful_partner = np.random.uniform(0, 1)
        init_stickiness_own = np.random.uniform(0, 1)
        init_stickiness_partner = np.random.uniform(0, 1)

        return np.array(
            [
                init_lr_own,
                init_lr_partner,
                init_beta,
                init_coef_info_bonus,
                init_forgetful_own,
                init_forgetful_partner,
                init_stickiness_own,
                init_stickiness_partner,
            ]
        )

    def constraints(self):
        """
        Define constraints for the optimization problem.

        Returns:
            constraints: LinearConstraint
                Constraints for the optimizer.
        """
        n_params = 8  # Number of parameters to estimate
        A = np.eye(n_params)
        lb = [0] * n_params  # Lower bounds for parameters
        ub = [
            1,  # lr_own
            1,  # lr_partner
            np.inf,  # beta
            1,  # coef_info_bonus
            1,  # forgetful_own
            1,  # forgetful_partner
            1,  # stickiness_own
            1,  # stickiness_partner
        ]
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


class HierarchicalBayesianStickyQSoftmaxWithOwnReward(
    HierarchicalBayesianQSoftmaxWithOwnReward
):
    def __init__(self):
        super().__init__()
        module_path = os.path.dirname(__file__)
        self.stan_file = os.path.join(
            module_path,
            "stan_files/hierarchical_social_sticky_q_learning_with_own_rewards.stan",
        )
        self.group2ind = None


class HierarchicalBayesianStickyQSoftmaxInfoBonusWithOwnReward(
    HierarchicalBayesianQSoftmaxWithOwnReward
):
    def __init__(self):
        super().__init__()
        module_path = os.path.dirname(__file__)
        self.stan_file = os.path.join(
            module_path,
            "stan_files/hierarchical_social_sticky_q_learning_bonus_term_with_own_rewards.stan",
        )
        self.group2ind = None


class HierarchicalBayesianStickyForgetfulQSoftmaxWithOwnReward(
    HierarchicalBayesianQSoftmaxWithOwnReward
):
    def __init__(self):
        super().__init__()
        module_path = os.path.dirname(__file__)
        self.stan_file = os.path.join(
            module_path,
            "stan_files/hierarchical_social_sticky_forgetful_q_learning_with_own_rewards.stan",
        )
        self.group2ind = None


class HierarchicalBayesianForgetfulQSoftmaxInfoBonusWithOwnReward(
    HierarchicalBayesianQSoftmaxWithOwnReward
):
    def __init__(self):
        super().__init__()
        module_path = os.path.dirname(__file__)
        self.stan_file = os.path.join(
            module_path,
            "stan_files/hierarchical_social_forgetful_q_learning_bonus_term_with_own_rewards.stan",
        )
        self.group2ind = None


class HierarchicalBayesianStickyForgetfulQSoftmaxInfoBonusWithOwnReward(
    HierarchicalBayesianQSoftmaxWithOwnReward
):
    def __init__(self):
        super().__init__()
        module_path = os.path.dirname(__file__)
        self.stan_file = os.path.join(
            module_path,
            "stan_files/hierarchical_social_sticky_forgetful_q_learning_bonus_term_with_own_rewards.stan",
        )
        self.group2ind = None


class HierarchicalBayesianWithinSubjectStickyForgetfulQSoftmaxWithOwnReward:
    def __init__(self):
        super().__init__()
        module_path = os.path.dirname(__file__)
        self.stan_file = os.path.join(
            module_path,
            "stan_files/hierarchical_social_within_subject_sticky_forgetful_q_learning_with_own_rewards.stan",
        )
        self.group2ind = None

    def fit(self, df):
        from cmdstanpy import CmdStanModel

        model = CmdStanModel(stan_file=self.stan_file)
        stan_data = self.convert_stan_data(df)
        self.posterior_sample = model.sample(data=stan_data)

    def convert_stan_data(self, df):
        from sklearn.preprocessing import LabelEncoder

        """
        Converts a pandas DataFrame into a Stan-compatible data dictionary.
    
        Parameters:
        - df (pd.DataFrame): DataFrame containing the experiment data with columns:
            'participant_id', 'session', 'trial', 'choice', 'reward',
            'partner_choice', 'partner_reward', 'condition'
    
        Returns:
        - stan_data (dict): Dictionary formatted for Stan model input.
        """

        # Step 1: Data Cleaning
        df = df.dropna(
            subset=[
                "participant_id",
                "session",
                "trial",
                "choice",
                "reward",
                "partner_choice",
                "partner_reward",
                "condition",  # Ensure 'condition' is included
            ]
        )

        # Step 2: Encode Choices
        # Check if 'choice' and 'partner_choice' are categorical strings
        if df["choice"].dtype == object or df["partner_choice"].dtype == object:
            choice_encoder = LabelEncoder()
            df["choice_encoded"] = choice_encoder.fit_transform(df["choice"]) + 1
            df["partner_choice_encoded"] = (
                choice_encoder.fit_transform(df["partner_choice"]) + 1
            )
        else:
            # If choices are numeric and start at 0, increment by 1 for Stan's 1-based indexing
            df["choice_encoded"] = df["choice"] + 1
            df["partner_choice_encoded"] = df["partner_choice"] + 1

        # Step 3: Determine Dimensions
        N = df["participant_id"].nunique()
        S = 4  # Number of sessions per participant is fixed at 2
        # Assuming each session has the same number of trials
        # Find the maximum number of trials across all sessions and participants
        T = df.groupby(["participant_id", "session"])["trial"].nunique().max()
        NC = max(df["choice_encoded"].max(), df["partner_choice_encoded"].max())

        print(f"\nNumber of Participants (N): {N}")
        print(f"Number of Sessions per Participant (S): {S}")
        print(f"Number of Trials per Session (T): {T}")
        print(f"Number of Choices/Actions (NC): {NC}\n")

        # Step 4: Initialize Arrays
        # Initialize zero-filled arrays
        C = np.zeros((N, S, T), dtype=int)  # Your own choices
        R = np.zeros((N, S, T), dtype=int)  # Your own rewards
        PC = np.zeros((N, S, T), dtype=int)  # Partner's choices
        PR = np.zeros((N, S, T), dtype=int)  # Partner's rewards
        condition = np.zeros((N, S), dtype=int)  # Condition indicator (1 = A, 2 = B)

        # Step 5: Sort Data for Consistent Ordering
        df_sorted = df.sort_values(by=["participant_id", "session", "trial"])

        # Step 6: Create Mappings
        participant_ids = sorted(df_sorted["participant_id"].unique())

        # Mapping: participant_id -> participant_index (0 to N-1)
        participant_mapping = {pid: idx for idx, pid in enumerate(participant_ids)}

        # Mapping: For each participant, map their unique session identifiers to session indices (0 and 1)
        # This handles arbitrary session labels per participant
        session_mapping = {}  # Dict of participant_id to {session_id: session_index}

        for pid in participant_ids:
            participant_sessions = sorted(
                df_sorted[df_sorted["participant_id"] == pid]["session"].unique()
            )
            if len(participant_sessions) != S:
                raise ValueError(
                    f"Participant {pid} does not have exactly {S} sessions."
                )
            # Assign session indices 0 and 1 based on sorted order
            session_mapping[pid] = {
                session_id: idx for idx, session_id in enumerate(participant_sessions)
            }

        # Step 7: Populate Arrays
        for _, row in df_sorted.iterrows():
            pid = row["participant_id"]
            p_idx = participant_mapping[pid]
            session_id = row["session"]
            # Get session index (0 or 1) for this participant
            s_idx = session_mapping[pid][session_id]
            t_idx = int(row["trial"]) - 1  # Adjust if trials start at 1

            # Safety check: Ensure trial index is within bounds
            if t_idx < 0 or t_idx >= T:
                raise ValueError(
                    f"Trial index out of bounds for participant {pid}, session {session_id}: trial {row['trial']}"
                )

            # Assign choices and rewards
            C[p_idx, s_idx, t_idx] = row["choice_encoded"]
            R[p_idx, s_idx, t_idx] = row["reward"]
            PC[p_idx, s_idx, t_idx] = row["partner_choice_encoded"]
            PR[p_idx, s_idx, t_idx] = row["partner_reward"]

            # Assign condition based on the 'condition' column
            # Ensure that 'condition' values are 1 or 2
            cond = row["condition"]
            if cond not in [1, 2]:
                raise ValueError(
                    f"Invalid condition value for participant {pid}, session {session_id}: {cond}"
                )
            condition[p_idx, s_idx] = cond

        # Step 8: Verify Shapes and Data Integrity
        print("C shape:", C.shape)  # (N, S, T)
        print("R shape:", R.shape)  # (N, S, T)
        print("PC shape:", PC.shape)  # (N, S, T)
        print("PR shape:", PR.shape)  # (N, S, T)
        print("condition shape:", condition.shape)  # (N, S)\n

        # Optional: Verify no zeros remain in choice arrays if choices are valid
        if np.any(C == 0):
            raise ValueError(
                "Zero entries found in C. Ensure that all choices are correctly encoded and present."
            )
        if np.any(PC == 0):
            raise ValueError(
                "Zero entries found in PC. Ensure that all partner choices are correctly encoded and present."
            )

        # Step 9: Prepare Stan Data Dictionary
        stan_data = {
            "N": N,
            "S": S,
            "T": T,
            "NC": NC,
            "C": C,
            "R": R,
            "PC": PC,
            "PR": PR,
            "condition": condition,
        }

        return stan_data
