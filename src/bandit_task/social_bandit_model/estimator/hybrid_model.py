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
            for unchosen_choice in range(self.num_choices):
                if unchosen_choice != current_partner_choice:
                    action_values[t, unchosen_choice] = action_values[
                        t - 1, unchosen_choice
                    ] + lr_for_action_values * (
                        0 - action_values[t - 1, unchosen_choice]
                    )

        # Calculate choice probabilities using softmax function
        combined_values = q_values * weights_for_values + action_values * (
            1 - weights_for_values
        )
        choice_prob = softmax(combined_values * beta, axis=1)

        # Calculate negative log-likelihood using your own choices not partners!
        chosen_prob = choice_prob[np.arange(1, n_trials), self.your_choices[:-1]]
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


class ForgetfulRewardActionHybridMLE(MLEstimator):
    def __init__(self) -> None:
        super().__init__()

    def neg_ll(self, params: Sequence[int | float]) -> float:
        (
            lr_for_q_values,
            lr_for_action_values,
            forget_rate,
            beta,
            weights_for_values,
        ) = params

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
                    q_values[t, unchosen_choice] = 0.5 * forget_rate + q_values[
                        t - 1, unchosen_choice
                    ] * (1 - forget_rate)

            # Action value update
            action_values[t, current_partner_choice] = action_values[
                t - 1, current_partner_choice
            ] + lr_for_action_values * (
                1 - action_values[t - 1, current_partner_choice]
            )

            # For actions not taken, action values are updated
            for unchosen_choice in range(self.num_choices):
                if unchosen_choice != current_partner_choice:
                    action_values[t, unchosen_choice] = action_values[
                        t - 1, unchosen_choice
                    ] + lr_for_action_values * (
                        0 - action_values[t - 1, unchosen_choice]
                    )

        # Calculate choice probabilities using softmax function
        combined_values = q_values * weights_for_values + action_values * (
            1 - weights_for_values
        )
        choice_prob = softmax(combined_values * beta, axis=1)

        # Calculate negative log-likelihood using your own choices not partners!
        chosen_prob = choice_prob[np.arange(1, n_trials), self.your_choices[:-1]]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        init_lr_for_q_values = np.random.beta(2, 2)
        init_lr_for_action_values = np.random.beta(2, 2)
        init_forget_rate = np.random.beta(2, 2)
        init_beta = np.random.gamma(2, 0.333)
        init_weights = np.random.beta(2, 2)
        return np.array(
            [
                init_lr_for_q_values,
                init_lr_for_action_values,
                init_forget_rate,
                init_beta,
                init_weights,
            ]
        )

    def constraints(self):
        A = np.eye(5)
        lb = np.array([0, 0, 0, 0, 0])
        ub = [1, 1, 1, np.inf, 1]
        return LinearConstraint(A, lb, ub)


class RewardImmediateActionHybridMLE(MLEstimator):
    def __init__(self) -> None:
        """
        The free parameters are
            - a learning rate for Q values `lr_for_q_values`
            - an inverse temperature for Q values `beta_for_q_values`.
            - an inverse temperature for action values `beta_for_action_values`
        """
        super().__init__()

    def neg_ll(self, params: Sequence[int | float]) -> float:
        lr_for_q_values, beta, stickiness = params

        # Initialize all the kind values matrix with 1/2s
        q_values = np.ones((len(self.your_choices), self.num_choices)) / 2
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

        # Calculate choice probabilities using softmax function
        combined_values = q_values.copy()
        combined_values[np.arange(1, n_trials), self.partner_choices[:-1]] += stickiness
        choice_prob = softmax(combined_values * beta, axis=1)

        # Calculate negative log-likelihood using your own choices not partners!
        chosen_prob = choice_prob[np.arange(1, n_trials), self.your_choices[:-1]]
        nll = -np.log(chosen_prob + 1e-8).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        init_lr_for_q_values = np.random.beta(2, 2)
        init_beta = np.random.gamma(2, 0.333)
        init_stickiness = np.random.beta(2, 2)
        return np.array([init_lr_for_q_values, init_beta, init_stickiness])

    def constraints(self):
        A = np.eye(3)
        lb = np.array([0, 0, 0])
        ub = [1, np.inf, 1]
        return LinearConstraint(A, lb, ub)


class HierarchicalBayesianRewardActionHybrid(HierarchicalEstimator):
    def __init__(self):
        super().__init__()
        module_path = os.path.dirname(__file__)
        self.stan_file = os.path.join(
            module_path,
            "stan_files/hierarchical_social_action_q_learning_without_rewards.stan",
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


class HierarchicalBayesianForgetfulRewardActionHybrid(
    HierarchicalBayesianRewardActionHybrid
):
    def __init__(self):
        super().__init__()
        module_path = os.path.dirname(__file__)
        self.stan_file = os.path.join(
            module_path,
            "stan_files/hierarchical_social_forgetful_action_q_learning_without_rewards.stan",
        )
        self.group2ind = None


class HierarchicalBayesianRewardImmediateActionHybrid(HierarchicalEstimator):
    def __init__(self):
        super().__init__()
        module_path = os.path.dirname(__file__)
        self.stan_file = os.path.join(
            module_path,
            "stan_files/hierarchical_social_immediate_action_q_learning_without_rewards.stan",
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


class RewardActionHybridMLEWithOwnReward(MLEstimator):
    """
    Maximum Likelihood Estimation for the RewardActionHybridSimulatorWithOwnReward model.
    Inherits methods from MLEstimator and implements session-specific negative log-likelihood.
    """

    def __init__(self, num_choices=2) -> None:
        """
        Initialize the MLE estimator.

        Parameters
        ----------
        num_choices : int, optional
            Number of possible choices/actions.
            Default is 2.
        """
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
        """
        Compute the negative log-likelihood for a single session.

        Parameters
        ----------
        params : Sequence[float]
            Array of parameters [lr_own, lr_partner_reward, lr_partner_action, beta, weights_for_value]
        your_choices : ndarray
            Participant's own choices (0-based indexing).
        your_rewards : ndarray
            Participant's own rewards.
        partner_choices : ndarray
            Partner's choices (0-based indexing).
        partner_rewards : ndarray
            Partner's rewards.

        Returns
        -------
        float
            Negative log-likelihood for the session.
        """
        lr_own, lr_partner_reward, lr_partner_action, beta, weights_for_value = params

        n_trials = len(your_choices)

        # Initialize Q-values and action values to 0.5 for all choices
        Q = np.ones(self.num_choices) / 2.0
        action_values = np.ones(self.num_choices) / self.num_choices

        total_neg_log_lik = 0.0

        for t in range(n_trials):
            # Compute combined values
            combined_values = (
                weights_for_value * Q + (1 - weights_for_value) * action_values
            )

            # Compute choice probabilities using softmax
            logits = beta * combined_values
            probs = softmax(logits)

            # Get participant's choice for this trial
            your_choice = your_choices[t]

            # Compute log probability of the chosen action
            prob_choice = probs[your_choice] + 1e-8  # Add epsilon to prevent log(0)
            log_prob = np.log(prob_choice)

            # Accumulate negative log-likelihood
            total_neg_log_lik -= log_prob

            # Update Q-values based on own reward
            your_reward = your_rewards[t]
            Q[your_choice] += lr_own * (your_reward - Q[your_choice])

            # Update Q-values and action values based on partner's choice and reward
            partner_choice = partner_choices[t]
            partner_reward = partner_rewards[t]

            # Update Q-values based on partner's reward
            Q[partner_choice] += lr_partner_reward * (
                partner_reward - Q[partner_choice]
            )

            # Update action values based on partner's action
            action_values[partner_choice] += lr_partner_action * (
                1 - action_values[partner_choice]
            )

            # Decay action values for unchosen actions
            for a in range(self.num_choices):
                if a != partner_choice:
                    action_values[a] += lr_partner_action * (0 - action_values[a])

        return total_neg_log_lik

    def initialize_params(self) -> np.ndarray:
        """
        Initialize parameters with random starting values within bounds.

        Returns
        -------
        ndarray
            Array of initial parameter values.
        """
        init_lr_own = np.random.uniform(0, 1)
        init_lr_partner_reward = np.random.uniform(0, 1)
        init_lr_partner_action = np.random.uniform(0, 1)
        init_beta = np.random.uniform(0.01, 100)
        init_weights_for_value = np.random.uniform(0, 1)
        return np.array(
            [
                init_lr_own,
                init_lr_partner_reward,
                init_lr_partner_action,
                init_beta,
                init_weights_for_value,
            ]
        )

    def constraints(self):
        """
        Define parameter constraints.

        Returns
        -------
        LinearConstraint
            Constraint object for scipy.optimize.minimize.
        """
        A = np.eye(5)
        lb = np.array([0, 0, 0, 0, 0])
        ub = np.array([1, 1, 1, np.inf, 1])
        return LinearConstraint(A, lb, ub)

    def get_param_names(self):
        """
        Get the names of the parameters.

        Returns
        -------
        list of str
            List of parameter names.
        """
        return [
            "lr_for_own_reward",
            "lr_for_partner_reward",
            "lr_for_partner_action",
            "beta",
            "weights_for_value",
        ]

    def print_results(self):
        """
        Print the estimated parameters in a readable format.
        """
        params_dict = {
            name: val for name, val in zip(self.get_param_names(), self.params)
        }
        print("\nEstimated Parameters:")
        print(
            f"  Learning Rate for Own Reward       : {params_dict['lr_for_own_reward']:.4f}"
        )
        print(
            f"  Learning Rate for Partner Reward   : {params_dict['lr_for_partner_reward']:.4f}"
        )
        print(
            f"  Learning Rate for Partner Action   : {params_dict['lr_for_partner_action']:.4f}"
        )
        print(f"  Beta (Softmax Temperature)         : {params_dict['beta']:.4f}")
        print(
            f"  Weights for Value                  : {params_dict['weights_for_value']:.4f}"
        )


class HierarchicalBayesianHybridSoftmaxWithOwnReward(HierarchicalEstimator):
    def __init__(
        self,
    ):
        super().__init__()
        module_path = os.path.dirname(__file__)
        self.stan_file = os.path.join(
            module_path,
            "stan_files/hierarchical_social_hybrid_learning_with_own_rewards.stan",
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


class WithinSubjectHierarchicalBayesianHybridSoftmaxWithOwnReward:
    def __init__(
        self,
    ):
        super().__init__()
        module_path = os.path.dirname(__file__)
        self.stan_file = os.path.join(
            module_path,
            "stan_files/within_subject_hierarchical_social_hybrid_learning_with_own_rewards.stan",
        )
        self.group2ind = None

    def fit(self, df):
        from cmdstanpy import CmdStanModel

        stan_data = self.convert_stan_data(df)
        model = CmdStanModel(stan_file=self.stan_file)
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
        S = 4  # Number of sessions per participant is fixed at 4 (2 conditions × 2 sessions)
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

        # Mapping: For each participant, map their unique session identifiers to session indices (0 to 3)
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
            # Assign session indices 0, 1, 2, 3 based on sorted order
            # Alternatively, map based on condition to ensure two sessions per condition
            # Here, we'll map based on sorted order
            session_mapping[pid] = {
                session_id: idx for idx, session_id in enumerate(participant_sessions)
            }

        # Step 7: Populate Arrays
        for _, row in df_sorted.iterrows():
            pid = row["participant_id"]
            p_idx = participant_mapping[pid]
            session_id = row["session"]
            # Get session index (0 to 3) for this participant
            s_idx = session_mapping[pid][session_id]
            t_idx = int(row["trial"])  # Adjust if trials start at 1

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
