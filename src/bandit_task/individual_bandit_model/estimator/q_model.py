import os

import numpy as np
from scipy.optimize import LinearConstraint
from scipy.special import softmax

from .base import MLEstimator, HierarchicalEstimator, HierarchicalWithinSubjectEstimator
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


class HierarchicalBayesianWithinSubjectQSoftmax(HierarchicalWithinSubjectEstimator):
    def __init__(self):
        super().__init__()
        module_path = os.path.dirname(__file__)
        self.stan_file = os.path.join(
            module_path, "stan_files/within_subject_hierarchical_q_learning.stan"
        )

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
                raise ValueError(f"Participant {pid} does not have exactly {S} sessions.")
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
