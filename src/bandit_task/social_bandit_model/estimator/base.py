import scipy.stats
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from cmdstanpy import CmdStanModel

from ...type import NDArrayNumber
from ...lib.utility import optimize_non_convex_obj
from typing import Sequence, Union
from numpy.typing import NDArray


class BaseEstimator(ABC):
    """
    The computational social bandit model base class.
    This serves as an abstract class for estimator classes for the social bandit models.
    """

    @abstractmethod
    def fit(
        self,
        num_choices: int,
        your_choices: Sequence[int | float],
        your_rewards: Sequence[int | float] | None,
        partner_choices: Sequence[int | float],
        partner_rewards: Sequence[int | float] | None,
        **kwargs: dict,
    ) -> None:
        """
        Abstract method for fitting the estimator.

        Parameters
        ----------
        num_choices : int
            The total number of choices available.
        your_choices : Sequence[int | float]
            The choices made by a user
        your_rewards : Sequence[int | float] | None
            The rewards observed by users. In some situations, the rewards of your own are not observable.
            In this case, set this argument to None.
        partner_choices : Sequence[int | float]
            The choices made by a partner.
        partner_rewards : Sequence[int | float]
            The rewards that a partner obtained. In some situations, the rewards of a partner are not observable or not used.
            In this case, set this argument to None.
        kwargs : dict
            Additional optional parameters for fitting.
        """
        pass

    def calculate_ic(self) -> float:
        """Calculate information criteria"""
        raise NotImplementedError


class MLEstimator(BaseEstimator):
    """
    Maximum Likelihood Estimator class extended to handle both single-session and multiple sessions.

    This class is responsible for finding the parameters that maximize
    the likelihood of the observed data across sessions.
    """

    def __init__(self):
        # Placeholder for the estimated parameters post-fitting
        self.estimated_params = None
        self.num_choices = None
        self.your_choices = None  # List of arrays, one per session
        self.your_rewards = None  # List of arrays, one per session
        self.partner_choices = None  # List of arrays, one per session
        self.partner_rewards = None  # List of arrays, one per session
        self.min_nll = None  # Minimized negative log-likelihood
        self.priors = None  # Prior

    def fit(
        self,
        num_choices: int,
        your_choices: Union[Sequence[int], Sequence[Sequence[int]]],
        your_rewards: Union[Sequence[float], Sequence[Sequence[float]]],
        partner_choices: Union[Sequence[int], Sequence[Sequence[int]]],
        partner_rewards: Union[Sequence[float], Sequence[Sequence[float]]],
        **kwargs,
    ) -> NDArray[np.float64]:
        """
        Fit the model using Maximum Likelihood Estimation.

        Parameters:
            num_choices: int
                The number of possible choices/actions.
            your_choices: Sequence of ints or sequences of ints
                Either a single array of your choices or a list of arrays for multiple sessions.
            your_rewards: Sequence of floats or sequences of floats
                Either a single array of your rewards or a list of arrays for multiple sessions.
            partner_choices: Sequence of ints or sequences of ints
                Either a single array of your partner's choices or a list of arrays for multiple sessions.
            partner_rewards: Sequence of floats or sequences of floats
                Either a single array of your partner's rewards or a list of arrays for multiple sessions.
            **kwargs: dict
                Additional keyword arguments for the optimizer.
        """
        # Check if inputs are single-session data (arrays) or multi-session data (lists)
        if isinstance(your_choices[0], (list, np.ndarray)):
            # Multi-session data
            num_sessions = len(your_choices)
            # Convert all inputs to lists of arrays
            self.your_choices = np.array(your_choices)
            self.your_rewards = np.array(your_rewards)
            self.partner_choices = np.array(partner_choices)
            self.partner_rewards = np.array(partner_rewards)
        else:
            # Single-session data
            num_sessions = 1
            # Wrap data in lists
            self.your_choices = np.array(your_choices).reshape(1, -1)
            self.your_rewards = np.array(your_rewards).reshape(1, -1)
            self.partner_choices = np.array(partner_choices).reshape(1, -1)
            self.partner_rewards = np.array(partner_rewards).reshape(1, -1)

        # Now proceed with data validation as before
        for i in range(num_sessions):
            if len(self.your_choices[i]) != len(self.your_rewards[i]):
                raise ValueError(
                    f"In session {i}, the sizes of your_choices and your_rewards must be the same."
                )
            if len(self.partner_choices[i]) != len(self.partner_rewards[i]):
                raise ValueError(
                    f"In session {i}, the sizes of partner_choices and partner_rewards must be the same."
                )
            if np.max(self.your_choices[i]) >= num_choices:
                raise ValueError(
                    f"In session {i}, the values in your_choices exceed num_choices."
                )
            if np.max(self.partner_choices[i]) >= num_choices:
                raise ValueError(
                    f"In session {i}, the values in partner_choices exceed num_choices."
                )

        self.num_choices = num_choices

        # Extract optimization options from keyword arguments
        options_for_min = kwargs.get("options", {})
        method = options_for_min.get(
            "method"
        )  # Use an optimizer that supports constraints
        n_trials = options_for_min.get(
            "n_trials", 5
        )  # Number of optimization runs to prevent local minima.

        # Initialize parameters
        initial_params = self.initialize_params()

        # Perform optimization
        self.min_nll, self.estimated_params = optimize_non_convex_obj(
            obj=self.neg_ll,
            init_param=initial_params,
            constraints=self.constraints(),
            method=method,
            n_trials=n_trials,
            options=options_for_min,
        )

        return self.estimated_params

    @abstractmethod
    def initialize_params(self) -> NDArray[np.float64]:
        """
        Abstract method for initializing parameters for optimization.
        """
        pass

    def neg_ll(self, params: Sequence[int | float]) -> float:
        """
        Calculate the negative log-likelihood for the current parameters.
        """
        total_neg_log_likelihood = 0.0

        # Loop over sessions
        for session_idx in range(len(self.your_choices)):
            your_choices = self.your_choices[session_idx]
            your_rewards = self.your_rewards[session_idx]
            partner_choices = self.partner_choices[session_idx]
            partner_rewards = self.partner_rewards[session_idx]

            total_neg_log_likelihood += self.session_neg_ll(
                params, your_choices, your_rewards, partner_choices, partner_rewards
            )

        # Add priors
        if self.priors is not None:
            for param, prior in zip(params, self.priors):
                if prior is not None:
                    total_neg_log_likelihood -= prior.logpdf(param)

        return total_neg_log_likelihood

    @abstractmethod
    def session_neg_ll(
        self, params, your_choices, your_rewards, partner_choices, partner_rewards
    ):
        """Calculate per session negative log-likelihood.
        Parameters
        ----------
        params
        your_choices
        your_rewards
        partner_choices
        partner_rewards

        Returns
        -------

        """
        pass

    @abstractmethod
    def constraints(self):
        """
        Define constraints for the optimization problem.
        """
        pass


class BayesianEstimator:
    """
    Bayesian model estimator.
    """

    def __init__(self):
        self.stan_file = None
        self.posterior_sample = None

    def fit(
        self,
        num_choices: int,
        your_choices: Sequence[int | float],
        your_rewards: Sequence[int | float] | None,
        partner_choices: Sequence[int | float],
        partner_rewards: Sequence[int | float] | None,
    ) -> None:
        """
        Fit the Bayesian model to the provided data.

        Parameters
        ----------
        num_choices : int
            The total number of choices available.
        your_choices : Sequence[int | float]
            The choices made by a user
        your_rewards : Sequence[int | float] | None
            The rewards observed by users. In some situations, the rewards of your own are not observable.
            In this case, set this argument to None.
        partner_choices : Sequence[int | float]
            The choices made by a partner.
        partner_rewards : Sequence[int | float]
            The rewards that a partner obtained. In some situations, the rewards of a partner are not observable or not used.
            In this case, set this argument to None.
        Returns
        -------
        None
            The results are stored as attributes of the instance.
        """
        your_choices = np.array(your_choices)
        if your_rewards is not None:
            your_rewards = np.array(your_rewards)
        partner_choices = np.array(partner_choices)
        if partner_rewards is not None:
            partner_rewards = np.array(partner_rewards)

        if your_choices.shape != partner_choices.shape:
            raise ValueError(
                f"The shapes of your_choices and partner_choices must match. "
                f"your_choices.shape={your_choices.shape} and partner_choices.shape={partner_choices.shape}"
            )

        if (your_rewards is not None) and (your_choices.shape != your_rewards.shape):
            raise ValueError(
                f"The shapes of your_choices and your_rewards must match. "
                f"your_choices.shape={your_choices.shape} and your_rewards.shape={your_rewards.shape}"
            )

        if (partner_rewards is not None) and (
            partner_choices.shape != partner_rewards.shape
        ):
            raise ValueError(
                f"The shapes of partner_choices and partner_rewards must match. "
                f"partner_rewards.shape={partner_choices.shape} and partner_rewards.shape={partner_rewards.shape}"
            )

        stan_data = self.convert_stan_data(
            num_choices, your_choices, your_rewards, partner_choices, partner_rewards
        )
        model = CmdStanModel(stan_file=self.stan_file)
        self.posterior_sample = model.sample(data=stan_data)

    @abstractmethod
    def convert_stan_data(
        self,
        num_choices: int,
        your_choices: NDArrayNumber,
        your_rewards: NDArrayNumber | None,
        partner_choices: NDArrayNumber,
        partner_rewards: NDArrayNumber | None,
    ) -> dict:
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
        your_choices: Sequence[int | float],
        your_rewards: Sequence[int | float] | None,
        partner_choices: Sequence[int | float],
        partner_rewards: Sequence[int | float] | None,
        groups: Sequence[int],
    ) -> None:
        """
        Fit the hierarchical model to the provided data.

        The method estimates both global and group-specific parameters.

        Parameters
        ----------
        num_choices : int
            The total number of choices available.
        your_choices : Sequence[int | float]
            The choices made by a user
        your_rewards : Sequence[int | float] | None
            The rewards observed by users. In some situations, the rewards of your own are not observable.
            In this case, set this argument to None.
        partner_choices : Sequence[int | float]
            The choices made by a partner.
        partner_rewards : Sequence[int | float]
            The rewards that a partner obtained. In some situations, the rewards of a partner are not observable or not used.
            In this case, set this argument to None.
        groups : Sequence[int | float]
            Group identifiers for each observation. Used to determine which observations belong to which group.

        Returns
        -------
        None
            The results are stored as attributes of the instance.

        Notes
        -----
        The exact hierarchical structure and which parameters are considered global vs.
        group-specific will depend on the implementation details.
        """
        your_choices = np.array(your_choices)
        if your_rewards is not None:
            your_rewards = np.array(your_rewards)
        partner_choices = np.array(partner_choices)
        if partner_rewards is not None:
            partner_rewards = np.array(partner_rewards)
        groups = np.array(groups)

        if your_choices.shape != partner_choices.shape:
            raise ValueError(
                f"The shapes of your_choices and partner_choices must match. "
                f"your_choices.shape={your_choices.shape} and partner_choices.shape={partner_choices.shape}"
            )

        if (your_rewards is not None) and (your_choices.shape != your_rewards.shape):
            raise ValueError(
                f"The shapes of your_choices and your_rewards must match. "
                f"your_choices.shape={your_choices.shape} and your_rewards.shape={your_rewards.shape}"
            )

        if (partner_rewards is not None) and (
            partner_choices.shape != partner_rewards.shape
        ):
            raise ValueError(
                f"The shapes of partner_choices and partner_rewards must match. "
                f"partner_rewards.shape={partner_choices.shape} and partner_rewards.shape={partner_rewards.shape}"
            )

        stan_data = self.convert_stan_data(
            num_choices,
            your_choices,
            your_rewards,
            partner_choices,
            partner_rewards,
            groups,
        )
        model = CmdStanModel(stan_file=self.stan_file)
        self.posterior_sample = model.sample(data=stan_data)

    @abstractmethod
    def convert_stan_data(
        self,
        num_choices: int,
        your_choices: NDArrayNumber,
        your_rewards: NDArrayNumber | None,
        partner_choices: NDArrayNumber,
        partner_rewards: NDArrayNumber | None,
        groups: NDArrayNumber,
    ) -> NDArrayNumber:
        pass

    def calculate_ic(self) -> float:
        """Calculate WAIC. The lower, the better"""
        log_lik = self.posterior_sample.stan_variable("log_lik")

        # Use log-sum-exp trick for numerical stability
        max_log_lik = np.max(log_lik, axis=0)
        lppd = (np.log(np.exp(log_lik - max_log_lik).mean(axis=0)) + max_log_lik).sum()

        penalty = np.sum(np.var(log_lik, axis=0))

        return -2 * (lppd - penalty)

    def estimate(self, variable_name, mode="mean"):
        if self.posterior_sample is None:
            raise Exception("Not fitted yet")

        if mode == "mean":
            return np.mean(self.posterior_sample.stan_variable(variable_name)), np.std(
                self.posterior_sample.stan_variable(variable_name)
            )

        if mode == "med":
            return np.mean(self.posterior_sample.stan_variable(variable_name))


class HierarchicalWithinSubjectEstimator:
    def __init__(self, seed=None):
        self.seed = seed

    def fit(self, df):
        from cmdstanpy import CmdStanModel

        model = CmdStanModel(stan_file=self.stan_file)
        stan_data = self.convert_stan_data(df)
        self.posterior_sample = model.sample(data=stan_data, seed=self.seed)

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

    def calculate_waic(self):
        """Calculate WAIC. Lower better"""
        log_lik = self.posterior_sample.stan_variable("log_lik")
        lppd = np.log(np.exp(log_lik).mean(axis=0)).sum()
        penalty = np.sum(np.var(log_lik, axis=0))
        return -2 * (lppd - penalty)
