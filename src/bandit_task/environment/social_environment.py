import copy
from warnings import warn

from ..bandit_instance.instance import Bandit
from ..individual_bandit_model.simulator.base import BaseSimulator
from ..lib.utility import check_params_type
from ..social_bandit_model.estimator.base import BaseEstimator


class Generator:
    """
    A class that simulates trials in a bandit environment using a specific simulator model.

    Attributes
    ----------
    simulator : BaseSimulator
        The simulator that decides which choices to make.
    partner : BaseSimulator
        The partner simulator
    original_simulator : BaseSimulator
        A deep copy of the original simulator to reset back to initial conditions.
    original_partner : BaseSimulator
        A deep copy of the original partner simulator to reset back to initial conditions.
    bandit_instance : Bandit
        The bandit instance that provides rewards for simulator's choices.
    done_simulation : bool
        Flag indicating if a simulation has been done.
    history : dict
        Contains the history of choices and corresponding rewards.
    total_trials : int
        Keeps track of total trials simulated so far.
    """

    def __init__(self, simulator: BaseSimulator, partner: BaseSimulator, bandit_instance: Bandit) -> None:
        """
        Initialize the generator with a simulator and bandit instance.

        Parameters
        ----------
        simulator : BaseSimulator
            The simulator model to use for making choices.
        partner : BaseSimulator
            The simulator model to generate observed choices and rewards
        bandit_instance : Bandit
            The bandit task to get rewards from based on choices.
        """
        check_params_type({simulator: BaseSimulator, partner: BaseSimulator, bandit_instance: Bandit})

        self.simulator = None
        self.partner = None
        self.history = None
        self.done_simulation = None
        self.total_trials = None
        self.original_simulator = copy.deepcopy(simulator)
        self.original_partner = copy.deepcopy(partner)
        self.bandit_instance = bandit_instance
        self.reset()

    def simulate(self, n_trials: int) -> None:
        """
        Simulate a given number of trials in the bandit environment.

        Parameters
        ----------
        n_trials : int
            The number of trials to simulate.
        """
        if self.done_simulation:
            warn(
                "The simulator has been already used to generate the data before. "
                "Which means the values of the simulator has changed from the initial values. "
                "If you do not mean to use the learnt simulator, use reset method."
            )
        for _ in range(n_trials):
            # Observing partner's choice and reward
            partner_choice = self.partner.make_choice()
            self.history["partner_choices"].append(partner_choice)
            partner_reward = self.bandit_instance.select_choice(partner_choice)
            self.history["partner_rewards"].append(partner_reward)

            self.partner.learn(partner_choice, partner_reward)
            self.simulator.learn(partner_choice, partner_reward)

            # Make your own choice
            your_choice = self.simulator.make_choice()
            self.history["your_choices"].append(your_choice)
            your_reward = self.bandit_instance.select_choice(your_choice)
            self.history["your_rewards"].append(your_reward)

        self.total_trials += n_trials
        self.done_simulation = True

    def reset(self):
        """
        Reset the simulator back to its initial conditions and clear the history.
        """
        self.simulator = copy.deepcopy(self.original_simulator)
        self.partner = copy.deepcopy(self.original_partner)
        self.history = {"your_choices": [], "your_rewards": [], "partner_choices": [], "partner_rewards": []}
        self.done_simulation = False
        self.total_trials = 0


class ParameterRecovery:

    def __init__(self, simulator: BaseSimulator, estimator: BaseEstimator, partner: BaseSimulator, bandit_instance: Bandit) -> None:
        check_params_type({simulator: BaseSimulator,
                           estimator: BaseEstimator,
                           partner: BaseSimulator,
                           bandit_instance: Bandit})

        self.estimator = estimator
        self.generator = Generator(simulator, partner, bandit_instance)

    def simulate(self, n_trials: int) -> None:
        self.generator.simulate(n_trials)

    def fit(self, **kwargs):
        num_choices = len(self.generator.bandit_instance.arms)
        self.estimator.fit(num_choices, **self.generator.history)