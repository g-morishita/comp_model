import copy
from warnings import warn

from ..bandit_instance.instance import Bandit
from ..individual_model.simulator.base import BaseSimulator


class Generator:
    """
    A class that simulates trials in a bandit environment using a specific simulator model.

    Attributes
    ----------
    simulator : BaseSimulator
        The simulator that decides which choices to make.
    original_simulator : BaseSimulator
        A deep copy of the original simulator to reset back to initial conditions.
    bandit_instance : Bandit
        The bandit instance that provides rewards for simulator's choices.
    done_simulation : bool
        Flag indicating if a simulation has been done.
    history : dict
        Contains the history of choices and corresponding rewards.
    total_trials : int
        Keeps track of total trials simulated so far.
    """

    def __init__(self, simulator: BaseSimulator, bandit_instance: Bandit) -> None:
        """
        Initialize the generator with a simulator and bandit instance.

        Parameters
        ----------
        simulator : BaseSimulator
            The simulator model to use for making choices.
        bandit_instance : Bandit
            The bandit task to get rewards from based on choices.
        """
        if not isinstance(simulator, BaseSimulator):
            raise ValueError(
                f"A simulator should be inherited from BaseSimulator class "
                f'from "simulator" package. {simulator.__class__.__name__} is given.'
            )
        if not isinstance(bandit_instance, Bandit):
            raise ValueError(
                f"bandit_task should be inherited Bandit class {bandit_instance.__class__.__name__} is given."
            )

        self.simulator = simulator
        self.original_simulator = copy.deepcopy(simulator)
        self.bandit_instance = bandit_instance
        self.done_simulation = False
        self.history = {"choices": [], "rewards": []}
        self.total_trials = 0

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
                "The simulator has been already used to generate the data before."
                "Which means the values of the simulator has changed from the initial values."
                "If you do not mean to use the learnt simulator, use reset method."
            )
        for _ in range(n_trials):
            choice = self.simulator.make_choice()
            self.history["choices"].append(choice)
            reward = self.bandit_instance.select_choice(choice)
            self.history["rewards"].append(reward)

            self.simulator.learn(choice, reward)

        self.total_trials += n_trials
        self.done_simulation = True

    def reset(self):
        """
        Reset the simulator back to its initial conditions and clear the history.
        """
        self.simulator = copy.deepcopy(self.original_simulator)
        self.history = {"choices": [], "rewards": []}
        self.done_simulation = False
        self.total_trials = 0
