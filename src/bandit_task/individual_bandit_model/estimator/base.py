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
