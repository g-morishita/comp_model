from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from cmdstanpy import CmdStanModel

# Custom imports from the parent directories
from ...lib.utility import read_options, optimize_non_convex_obj
from ...type import NDArrayNumber


class BaseEstimator(ABC):
    """
    The computational individual_bandit_model base class.
    This serves as an abstract class for other estimator classes.
    """

    @abstractmethod
    def fit(
        self,
        num_choices: int,
        choices: Sequence[int | float],
        rewards: Sequence[int | float],
        **kwargs: dict,
    ) -> Sequence[int | float]:
        """
        Abstract method for fitting the estimator.

        Parameters
        ----------
        num_choices : int
            The total number of choices available.
        choices : array_like
            The observed choices made by users or agents.
        rewards : array_like
            The rewards or outcomes corresponding to each action.
        kwargs : dict, optional
            Additional optional parameters for fitting.
        """
        pass


class MLEstimator(BaseEstimator):
    """
    Maximum Likelihood Estimator class.

    This class is responsible for finding the parameters that maximize
    the likelihood of the observed data.
    """

    def __init__(self):
        # Placeholder for the estimated parameters post-fitting
        self.estimated_params = None
        self.num_choices = None
        self.choices = None
        self.rewards = None

    def fit(
        self,
        num_choices: int,
        choices: Sequence[int],
        rewards: Sequence[int | float],
        **kwargs: dict,
    ) -> Sequence[int | float]:
        """
        Fit the model using Maximum Likelihood Estimation.
        """
        if len(choices) != len(rewards):
            raise ValueError("The sizes of `choices` and `rewards` must be the same.")
        if max(choices) > num_choices:
            raise ValueError("The range of `choices` exceeds `num_choices`.")

        self.num_choices = num_choices
        self.choices = np.array(choices)
        self.rewards = np.array(rewards)

        # Extract optimization options from keyword arguments
        options_for_min = read_options({"maxiter", "tol", "method", "n_trials"})
        method = options_for_min.get("method")
        n_trials = options_for_min.get(
            "n_trials", 5
        )  # The number of optimization to run to prevent local minima.

        self.estimated_params = optimize_non_convex_obj(
            self.neg_ll,
            self.initialize_params(),
            method=method,
            constraints=self.constraints(),
            n_trials=n_trials,
            options=options_for_min,
        )
        return self.estimated_params

    @abstractmethod
    def initialize_params(self) -> NDArrayNumber:
        """
        Abstract method for initializing parameters for optimization.
        """
        pass

    @abstractmethod
    def neg_ll(self, params: Sequence[int | float]) -> float:
        """
        Calculate the negative log-likelihood for the current parameters.
        """
        pass

    @abstractmethod
    def constraints(self):
        """
        Define constraints for the optimization problem.
        """
        pass


class BayesianEstimator(BaseEstimator):
    """
    Bayesian model estimator

    """

    def __init__(self):
        self.stan_file = None
        self.posterior_sample = None

    def fit(
        self,
        num_choices: int,
        choices: Sequence[int],
        rewards: Sequence[int | float],
        **kwargs: dict,
    ) -> None:
        """
        Fit the Bayesian model to the provided data.
        """
        if len(choices) != len(rewards):
            raise ValueError("The sizes of `choices` and `rewards` must be the same.")
        if max(choices) > num_choices:
            raise ValueError("The range of `choices` exceeds `num_choices`.")

        choices = np.array(choices)
        rewards = np.array(rewards)

        stan_data = self.convert_stan_data(num_choices, choices, rewards)
        model = CmdStanModel(stan_file=self.stan_file)
        self.posterior_sample = model.sample(data=stan_data)

    @abstractmethod
    def convert_stan_data(
        self,
        num_choices: int,
        choices: NDArrayNumber,
        rewards: NDArrayNumber,
    ) -> NDArrayNumber:
        pass


class HierarchicalEstimator:
    """
    Hierarchical (or multi-level) model estimator.

    This class implements a hierarchical model that estimates parameters at multiple
    levels. At the higher level, global parameters are estimated that apply across all groups.
    At the lower level, group-specific parameters are estimated that can vary from one group to another.

    A hierarchical model shares statistical strength across groups, making it especially
    useful when some groups might have limited data.

    Methods
    -------
    fit :
        Estimate the model's parameters based on the provided data.
    """

    def __init__(self):
        self.stan_file = None
        self.posterior_sample = None

    def fit(
        self,
        num_choices: int,
        choices: NDArrayNumber,
        rewards: NDArrayNumber,
        groups: NDArrayNumber,
        **kwargs: dict,
    ) -> None:
        """
        Fit the hierarchical model to the provided data.

        The method estimates both global and group-specific parameters.

        Parameters
        ----------
        num_choices : int
            The total number of choices available.
        choices : Sequence[int | float]
            The observed choices made by users or agents.
            The rows correspond to a session.
            The columns correspond to a trial.
        rewards : Sequence[int | float]
            The rewards or outcomes corresponding to each action.
            The rows correspond to a session.
            The columns correspond to a trial.
        groups : Sequence[int | float]
            Group identifiers for each observation. Used to determine which observations belong to which group.
        kwargs : dict, optional
            Additional optional parameters for fitting.

        Returns
        -------
        None
            The results are stored as attributes of the instance.

        Notes
        -----
        The exact hierarchical structure and which parameters are considered global vs.
        group-specific will depend on the implementation details.
        """
        choices = np.array(choices)
        rewards = np.array(rewards)
        groups = np.array(groups)
        if choices.shape != rewards.shape:
            raise ValueError(
                f"The shapes of choices and rewards must match. "
                f"choices.shape={choices.shape} and rewards.shape={rewards.shape}"
            )

        stan_data = self.convert_stan_data(num_choices, choices, rewards, groups)
        model = CmdStanModel(stan_file=self.stan_file)
        self.posterior_sample = model.sample(data=stan_data)

    @abstractmethod
    def convert_stan_data(
        self,
        num_choices: int,
        choices: Sequence[int | float],
        rewards: Sequence[int | float],
        groups: Sequence[int | float],
    ) -> NDArrayNumber:
        pass

    def calculate_waic(self):
        """Calculate WAIC"""
        log_lik = self.posterior_sample.stan_variable("log_lik")
        lppd = np.log(np.exp(log_lik).mean(axis=0)).sum()
        penalty = np.sum(np.var(log_lik, axis=0))
        return -2 * (lppd - penalty)


class HierarchicalWithinSubjectEstimator:
    def __init__(self):
        self.stan_file = None

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
                "condition",  # Ensure 'condition' is included
            ]
        )

        # Step 2: Encode Choices
        # Check if 'choice' and 'partner_choice' are categorical strings
        if df["choice"].dtype == object:
            choice_encoder = LabelEncoder()
            df["choice_encoded"] = choice_encoder.fit_transform(df["choice"]) + 1
        else:
            # If choices are numeric and start at 0, increment by 1 for Stan's 1-based indexing
            df["choice_encoded"] = df["choice"] + 1

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
            t_idx = int(row["trial"]) - 1  # Adjust if trials start at 1

            # Safety check: Ensure trial index is within bounds
            if t_idx < 0 or t_idx >= T:
                raise ValueError(
                    f"Trial index out of bounds for participant {pid}, session {session_id}: trial {row['trial']}"
                )

            # Assign choices and rewards
            C[p_idx, s_idx, t_idx] = row["choice_encoded"]
            R[p_idx, s_idx, t_idx] = row["reward"]

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
        print("condition shape:", condition.shape)  # (N, S)\n

        # Optional: Verify no zeros remain in choice arrays if choices are valid
        if np.any(C == 0):
            raise ValueError(
                "Zero entries found in C. Ensure that all choices are correctly encoded and present."
            )

        # Step 9: Prepare Stan Data Dictionary
        stan_data = {
            "N": N,
            "S": S,
            "T": T,
            "NC": NC,
            "C": C,
            "R": R,
            "condition": condition,
        }

        return stan_data

    def calculate_waic(self):
        """Calculate WAIC"""
        log_lik = self.posterior_sample.stan_variable("log_lik")
        lppd = np.log(np.exp(log_lik).mean(axis=0)).sum()
        penalty = np.sum(np.var(log_lik, axis=0))
        return -2 * (lppd - penalty)
