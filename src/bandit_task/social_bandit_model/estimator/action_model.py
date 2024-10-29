import os
from warnings import warn

import numpy as np
from scipy.optimize import LinearConstraint
from scipy.special import softmax

from .base import MLEstimator, HierarchicalEstimator
from typing import Sequence
from ...type import NDArrayNumber


class ActionSoftmaxMLEWithoutYourReward(MLEstimator):
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
        lr, beta = params

        # Initialize all the kind values matrix with 1/2s
        action_values = np.ones((len(your_choices), self.num_choices)) / 3
        n_trials = len(your_choices)

        # For each trial, calculate delta and update Q-values
        for t in range(1, n_trials):
            current_partner_choice = partner_choices[t - 1]  # Choice made at time t
            # Action value update
            action_values[t, current_partner_choice] = action_values[
                t - 1, current_partner_choice
            ] + lr * (1 - action_values[t - 1, current_partner_choice])

            # For actions not taken, action values are updated
            for unchosen_choice in range(self.num_choices):
                if unchosen_choice != current_partner_choice:
                    action_values[t, unchosen_choice] = action_values[
                        t - 1, unchosen_choice
                    ] + lr * (0 - action_values[t - 1, unchosen_choice])

        # Calculate choice probabilities using softmax function
        choice_prob = softmax(action_values * beta, axis=1)

        # Calculate negative log-likelihood using your own choices not partners!
        chosen_prob = choice_prob[np.arange(1, n_trials), your_choices[:-1]]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        lr = np.random.beta(2, 2)
        init_beta = np.random.gamma(2, 0.333)
        return np.array([lr, init_beta])

    def constraints(self):
        A = np.eye(2)
        lb = np.array([0, 0])
        ub = [1, np.inf]
        return LinearConstraint(A, lb, ub)


class ActionSoftmaxMLEWithYourReward(MLEstimator):
    def __init__(self, num_choices=3) -> None:
        super().__init__()
        self.num_choices = num_choices

    def session_neg_ll(
        self,
        params: Sequence[float],
        your_choices,
        your_rewards,
        partner_choices,
        partner_rewards,
    ) -> float:
        lr, beta = params
        n_trials = len(your_choices)

        # Initialize action values to 1/num_choices
        action_values = np.ones(self.num_choices) / self.num_choices

        total_neg_log_lik = 0.0

        for t in range(n_trials):
            # Compute choice probabilities using current action values
            logits = beta * action_values
            probs = softmax(logits)

            # Get participant's choice for this trial
            your_choice = your_choices[t]

            # Compute log probability of the chosen action
            prob_choice = probs[your_choice] + 1e-8  # Add epsilon to prevent log(0)
            log_prob = np.log(prob_choice)

            # Accumulate negative log-likelihood
            total_neg_log_lik -= log_prob

            # Update action values based on partner's choice at trial t
            partner_choice = partner_choices[t]

            # Update action values (only based on partner's choice)
            action_values[partner_choice] += lr * (1 - action_values[partner_choice])

            # Decay action values for unchosen actions
            for unchosen_choice in range(self.num_choices):
                if unchosen_choice != partner_choice:
                    action_values[unchosen_choice] += lr * (
                        0 - action_values[unchosen_choice]
                    )

        return total_neg_log_lik

    def initialize_params(self) -> np.ndarray:
        lr = np.random.uniform(0, 1)
        init_beta = np.random.uniform(0.01, 100)
        return np.array([lr, init_beta])

    def constraints(self):
        A = np.eye(2)
        lb = np.array([0, 0])
        ub = np.array([1, np.inf])
        return LinearConstraint(A, lb, ub)


class HierarchicalActionSoftmaxWithoutYourReward(HierarchicalEstimator):
    """
    Implements a bayesian hierarchical action learning model.

    """

    def __init__(self):
        super().__init__()
        module_path = os.path.dirname(__file__)
        self.stan_file = os.path.join(
            module_path, "stan_files/hierarchical_social_action_learning.stan"
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
    ):
        uniq_groups = np.unique(groups)

        if your_rewards is not None:
            warn(
                "`your_rewards` was given but it will not be used in this action learning model."
            )

        if partner_rewards is not None:
            warn(
                "`partner_rewards` was given but it will not be used in this action learning model."
            )

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

        self.group2ind = dict(zip(uniq_groups, np.arange(len(uniq_groups))))

        for g in uniq_groups:
            reshaped_your_choices[self.group2ind[g], :, :] = your_choices[groups == g]
            reshaped_partner_choices[self.group2ind[g], :, :] = partner_choices[
                groups == g
            ]

        stan_data = {
            "N": n_uniq_groups,
            "S": n_sessions_per_group,
            "T": n_trials,
            "NC": num_choices,
            "C": (reshaped_your_choices + 1).astype(int).tolist(),
            "PC": (reshaped_partner_choices + 1).astype(int).tolist(),
        }
        print(stan_data)
        return stan_data
