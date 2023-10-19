import warnings
from abc import ABC, abstractmethod
from typing import Sequence

import arviz as az
import numpy as np
import stan
from scipy.optimize import minimize

# Custom imports from the parent directories
from src.bandit_task.lib.utility import read_options
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
    ) -> None:
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
            raise ValueError("The sizes of `actions` and `rewards` must be the same.")
        if max(choices) > num_choices:
            raise ValueError("The range of `actions` exceeds `num_choices`.")

        self.num_choices = num_choices
        self.choices = np.array(choices)
        self.rewards = np.array(rewards)

        # Extract optimization options from keyword arguments
        options_for_min = read_options({"maxiter", "tol", "method", "n_trials"})
        method = options_for_min.get("method")
        n_trials = options_for_min.get("n_trials", 5)

        # Initialize variables for optimization results
        min_nll = np.inf  # Minimum negative log-likelihood
        opt_param = None  # Optimal parameters

        # Optimize using multiple initializations
        for _ in range(n_trials):
            init_param = self.initialize_params()
            result = minimize(
                self.neg_ll,
                init_param,
                method=method,
                constraints=self.constraints(),
                options=options_for_min,
            )

            # Warning if optimization was not successful
            if not result.success:
                warnings.warn(result.message)
            else:
                print("The minimization succeeded!")
                # Update the best parameters if new result is better
                if min_nll > result.fun:
                    min_nll = result.fun
                    opt_param = result.x

        if opt_param is None:
            warnings.warn("The estimation did not work.")
            return np.array([])

        self.estimated_params = opt_param
        return self.estimated_params

    @abstractmethod
    def initialize_params(self) -> Sequence[int | float]:
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


class HierarchicalEstimator:
    """
    Hierarchical (or multi-level) model estimator.

    This class implements a hierarchical model that estimates parameters at multiple
    levels. At the higher level, global parameters are estimated that apply across all groups.
    At the lower level, group-specific parameters are estimated that can vary from one group to another.

    A hierarchical model shares statistical strength across groups, making it especially
    useful when some groups might have limited data.

    Attributes
    ----------
    (Any class-level attributes should be documented here, if they exist.)

    Methods
    -------
    fit :
        Estimate the model's parameters based on the provided data.
    (Other methods should be documented similarly.)
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

        with open(self.stan_file, "r") as f:
            stan_code = f.read()

        stan_data = self.convert_stan_data(num_choices, choices, rewards, groups)

        posterior = stan.build(stan_code, data=stan_data, random_seed=1)
        posterior_fit = posterior.sample(num_chains=4, num_samples=1000)

        self.posterior_sample = az.from_pystan(
            posterior=posterior_fit,
            posterior_model=posterior,
            log_likelihood=["log_lik"],
        )

    @abstractmethod
    def convert_stan_data(
        self,
        num_choices: int,
        choices: Sequence[int | float],
        rewards: Sequence[int | float],
        groups: Sequence[int | float],
    ):
        pass
