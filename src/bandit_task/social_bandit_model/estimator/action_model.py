import os
from warnings import warn

import numpy as np
from scipy.optimize import LinearConstraint
from scipy.special import softmax

from .base import MLEstimator, HierarchicalEstimator
from ...type import NDArrayNumber


class BayesianActionSoftmaxWithoutYourReward:



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
            warn("`your_rewards` was given but it will not be used in this action learning model.")

        if partner_rewards is not None:
            warn("`partner_rewards` was given but it will not be used in this action learning model.")

        n_uniq_groups = uniq_groups.shape[0]
        # Assume every group has the same number of sessions.
        n_sessions_per_group = your_choices.shape[0] // n_uniq_groups
        n_trials = your_choices.shape[1]
        reshaped_your_choices = np.zeros((n_uniq_groups, n_sessions_per_group, n_trials))
        reshaped_partner_choices = np.zeros((n_uniq_groups, n_sessions_per_group, n_trials))

        self.group2ind = dict(zip(uniq_groups, np.arange(len(uniq_groups))))

        for g in uniq_groups:
            reshaped_your_choices[self.group2ind[g], :, :] = your_choices[groups == g]
            reshaped_partner_choices[self.group2ind[g], :, :] = partner_choices[groups == g]

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
