import os
from typing import Sequence

import numpy as np
from scipy.optimize import LinearConstraint
from scipy.special import softmax

from .base import MLEstimator, HierarchicalEstimator, BayesianEstimator
from ...type import NDArrayNumber


class RewardActionHybridMLE(MLEstimator):
    def __init__(self) -> None:
        """
        This class estimates free parameters of  a reward and action hybrid learning model, which learns from partner's choices and
        rewards and makes a choice using softmax function using the maximum likelihood estimator (MLE).
        The free parameters are
            - a learning rate for Q values `lr_for_q_values`
            - a learning rate for action values `lr_for_action_values`
            - an inverse temperature `beta`.
            - weights for q values `weights_for_values`
        """
        super().__init__()

    def neg_ll(self, params: Sequence[int | float]) -> float:
        lr_for_q_values, lr_for_action_values, beta, weights_for_values = params

        # Initialize all the kind values matrix with 1/2s
        q_values = np.ones((len(self.your_choices), self.num_choices)) / 2
        action_values = np.ones((len(self.your_choices), self.num_choices)) / 3
        n_trials = len(self.your_choices)

        # For each trial, calculate delta and update Q-values
        for t in range(1, n_trials):
            current_partner_choice = self.partner_choices[
                t - 1
            ]  # Choice made at time t
            current_partner_reward = self.partner_rewards[
                t - 1
            ]  # Reward received at time t
            delta_t = current_partner_reward - q_values[t - 1, current_partner_choice]

            # Q-value update
            q_values[t, current_partner_choice] = (
                q_values[t - 1, current_partner_choice] + lr_for_q_values * delta_t
            )
            # For actions not taken, Q-values remain the same
            for unchosen_choice in range(self.num_choices):
                if unchosen_choice != current_partner_choice:
                    q_values[t, unchosen_choice] = q_values[t - 1, unchosen_choice]

            # Action value update
            action_values[t, current_partner_choice] = action_values[
                t - 1, current_partner_choice
            ] + lr_for_action_values * (
                1 - action_values[t - 1, current_partner_choice]
            )

            # For actions not taken, action values are updated
            for unchosen_choice in range(len(self.your_choices)):
                if unchosen_choice != current_partner_choice:
                    action_values[unchosen_choice] = action_values[
                        unchosen_choice
                    ] + lr_for_action_values * (0 - action_values[unchosen_choice])

        # Calculate choice probabilities using softmax function
        combined_values = q_values * weights_for_values + action_values * (
            1 - weights_for_values
        )
        choice_prob = softmax(combined_values * beta, axis=1)

        # Calculate negative log-likelihood using your own choices not partners!
        chosen_prob = choice_prob[np.arange(n_trials), self.your_choices]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        init_lr_for_q_values = np.random.beta(2, 2)
        init_lr_for_action_values = np.random.beta(2, 2)
        init_beta = np.random.gamma(2, 0.333)
        init_weights = np.random.beta(2, 2)
        return np.array(
            [init_lr_for_q_values, init_lr_for_action_values, init_beta, init_weights]
        )

    def constraints(self):
        A = np.eye(4)
        lb = np.array([0, 0, 0, 0])
        ub = [1, 1, np.inf, 1]
        return LinearConstraint(A, lb, ub)


class RewardImmediateActionHybridMLE(MLEstimator):
    def __init__(self) -> None:
        """
        This class estimates free parameters of  a reward and action hybrid learning model, which learns from partner's choices and
        rewards and makes a choice using softmax function using the maximum likelihood estimator (MLE).
        The difference between `RewardActionHybridMLE` is the way of updating and holding action values.
        This class updates an action value for a choice that is just taken to 1 and others to 0.
        This is equivalent of fixing `lr_for_action` to 1.

        The free parameters are
            - a learning rate for Q values `lr_for_q_values`
            - an inverse temperature for Q values `beta_for_q_values`.
            - an inverse temperature for action values `beta_for_action_values`
        """
        super().__init__()

    def neg_ll(self, params: Sequence[int | float]) -> float:
        lr_for_q_values, beta_for_q_values, beta_for_action_values = params

        # Initialize all the kind values matrix with 1/2s
        q_values = np.ones((len(self.your_choices), self.num_choices)) / 2
        action_values = np.ones((len(self.your_choices), self.num_choices)) / 2
        n_trials = len(self.your_choices)

        # For each trial, calculate delta and update Q-values
        for t in range(1, n_trials):
            current_partner_choice = self.partner_choices[
                t - 1
            ]  # Choice made at time t
            current_partner_reward = self.partner_rewards[
                t - 1
            ]  # Reward received at time t
            delta_t = current_partner_reward - q_values[t - 1, current_partner_choice]

            # Q-value update
            q_values[t, current_partner_choice] = (
                q_values[t - 1, current_partner_choice] + lr_for_q_values * delta_t
            )
            # For actions not taken, Q-values remain the same
            for unchosen_choice in range(self.num_choices):
                if unchosen_choice != current_partner_choice:
                    q_values[t, unchosen_choice] = q_values[t - 1, unchosen_choice]

            # Action value update
            action_values[t, current_partner_choice] = 1

            # For actions not taken, action values are updated
            for unchosen_choice in range(len(self.your_choices)):
                if unchosen_choice != current_partner_choice:
                    action_values[unchosen_choice] = 0

        # Calculate choice probabilities using softmax function
        combined_values = (
            q_values * beta_for_q_values + action_values * beta_for_action_values
        )
        choice_prob = softmax(combined_values, axis=1)

        # Calculate negative log-likelihood using your own choices not partners!
        chosen_prob = choice_prob[np.arange(n_trials), self.your_choices]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        init_lr_for_q_values = np.random.beta(2, 2)
        init_beta_for_q_values = np.random.gamma(2, 0.333)
        init_beta_for_action_values = np.random.gamma(2, 0.333)
        return np.array(
            [init_lr_for_q_values, init_beta_for_q_values, init_beta_for_action_values]
        )

    def constraints(self):
        A = np.eye(3)
        lb = np.array([0, 0, 0])
        ub = [1, np.inf, np.inf]
        return LinearConstraint(A, lb, ub)
