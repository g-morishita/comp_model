from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from cmdstanpy import CmdStanModel

# Custom imports from the parent directories
from ...lib.utility import read_options, optimize_non_convex_obj
from ...type import NDArrayNumber


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
    Maximum Likelihood Estimator class.

    This class is responsible for finding the parameters that maximize
    the likelihood of the observed data.
    """

    def __init__(self):
        # Placeholder for the estimated parameters post-fitting
        self.estimated_params = None
        self.num_choices = None
        self.your_choices = None
        self.your_rewards = None
        self.partner_choices = None
        self.partner_rewards = None

    def fit(
        self,
        num_choices: int,
        your_choices: Sequence[int | float],
        your_rewards: Sequence[int | float] | None,
        partner_choices: Sequence[int | float],
        partner_rewards: Sequence[int | float] | None,
        **kwargs: dict,
    ) -> NDArrayNumber:
        """
        Fit the model using Maximum Likelihood Estimation.
        """
        if (your_rewards is not None) and len(your_choices) != len(your_rewards):
            raise ValueError(
                "The sizes of `your_choices` and `your_rewards` must be the same."
            )
        if (your_rewards is not None) and len(partner_choices) != len(partner_rewards):
            raise ValueError(
                "The sizes of `partner_choices` and `partner_rewards` must be the same."
            )
        if max(your_choices) > num_choices:
            raise ValueError("The range of `your_choices` exceeds `num_choices`.")
        if max(partner_choices) > num_choices:
            raise ValueError("The range of `your_choices` exceeds `num_choices`.")

        self.num_choices = num_choices
        self.your_choices = np.array(your_choices)
        self.your_rewards = your_rewards
        self.partner_choices = partner_choices
        self.partner_rewards = partner_rewards

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
        seed=None,
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
        seed:
            Seed for MCMC

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
        self.posterior_sample = model.sample(data=stan_data, seed=seed)

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
