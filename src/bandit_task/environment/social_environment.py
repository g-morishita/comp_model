import copy
from typing import Collection
from warnings import warn

from ..bandit_instance.instance import Bandit
from ..individual_bandit_model.simulator.base import BaseSimulator
from ..social_bandit_model.simulator.base import (
    BaseSimulator as SimulatorWithOwnRewards,
)
from ..lib.utility import check_params_type
from ..social_bandit_model.estimator.base import BaseEstimator, HierarchicalEstimator
from ..social_bandit_model.simulator.action_model import ActionSoftmaxSimulator


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

    def __init__(
        self, simulator: BaseSimulator, partner: BaseSimulator, bandit_instance: Bandit
    ) -> None:
        """
        Initialize the generator with a simulator and bandit instance.
        The data generation process is as follows:

        1. Partner Makes a Decision: In each round, the partner (another player or a computer program) decides on an action to take.
        2. Observe Partner's Reward: Based on this decision, a reward (like points or a score) is given.
        3. Partner Learns from Experience: The partner then uses the information from their decision and the received reward to improve their strategy for future rounds.
        4. Simulator Learns from Partner's Experience: A simulator, acting on your behalf, also learns from the partner's decision and reward.
        5. Simulator Makes Your Decision: Next, the simulator decides on an action for you.
        6. Observe Your Reward: A reward is calculated for your decision.

        Parameters
        ----------
        simulator : BaseSimulator
            The simulator model to use for making choices.
        partner : BaseSimulator
            The simulator model to generate observed choices and rewards
        bandit_instance : Bandit
            The bandit task to get rewards from based on choices.
        """
        check_params_type(
            {
                simulator: [BaseSimulator, ActionSoftmaxSimulator],
                partner: BaseSimulator,
                bandit_instance: Bandit,
            }
        )

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

            if isinstance(self.simulator, BaseSimulator):
                self.simulator.learn_from_partner(partner_choice, partner_reward)
            elif isinstance(self.simulator, ActionSoftmaxSimulator):
                self.simulator.learn(partner_choice)

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
        self.history = {
            "your_choices": [],
            "your_rewards": [],
            "partner_choices": [],
            "partner_rewards": [],
        }
        self.done_simulation = False
        self.total_trials = 0


class ParameterRecovery:
    def __init__(
        self,
        simulator: BaseSimulator,
        estimator: BaseEstimator,
        partner: BaseSimulator,
        bandit_instance: Bandit,
    ) -> None:
        check_params_type(
            {
                simulator: BaseSimulator,
                estimator: BaseEstimator,
                partner: BaseSimulator,
                bandit_instance: Bandit,
            }
        )

        self.estimator = estimator
        self.generator = Generator(simulator, partner, bandit_instance)

    def simulate(self, n_trials: int) -> None:
        self.generator.simulate(n_trials)

    def fit(self, **kwargs):
        num_choices = len(self.generator.bandit_instance.arms)
        self.estimator.fit(num_choices, **self.generator.history)


class GeneratorWithOwnRewards:
    """
    A class that simulates trials in a bandit environment using a specific simulator model.
    In this, simulator will observe its own reward outcome as well as the partner's.

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

    def __init__(
        self, simulator: BaseSimulator, partner: BaseSimulator, bandit_instance: Bandit
    ) -> None:
        """
        Initialize the generator with a simulator and bandit instance.
        The data generation process is as follows:

        1. Simulator Makes Your Decision: In each round, the simulator decides on an action.
        2. Simulator Learns from its Own Experience: A simulator, acting on your behalf, learns from its decision and reward.
        3. Partner Makes a Decision:  Next, the partner (another player or a computer program) decides on an action to take.
        4. Simulator Learns from Partner's Experience: A simulator, acting on your behalf, also learns from the partner's decision and reward.

        Parameters
        ----------
        simulator : BaseSimulator
            The simulator model to use for making choices.
        partner : BaseSimulator
            The simulator model to generate observed choices and rewards
        bandit_instance : Bandit
            The bandit task to get rewards from based on choices.
        """
        check_params_type(
            {
                simulator: [SimulatorWithOwnRewards],
                partner: SimulatorWithOwnRewards,
                bandit_instance: Bandit,
            }
        )

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
            # Make your own choice
            your_choice = self.simulator.make_choice()
            self.history["your_choices"].append(your_choice)
            your_reward = self.bandit_instance.select_choice(your_choice)
            self.history["your_rewards"].append(your_reward)
            self.simulator.learn_from_own(your_choice, your_reward)
            self.partner.learn_from_partner(your_choice, your_reward)

            # Partner's decision
            partner_choice = self.partner.make_choice()
            self.history["partner_choices"].append(partner_choice)
            partner_reward = self.bandit_instance.select_choice(partner_choice)
            self.history["partner_rewards"].append(partner_reward)
            self.partner.learn_from_own(partner_choice, partner_reward)
            self.simulator.learn_from_partner(partner_choice, partner_reward)

        self.total_trials += n_trials
        self.done_simulation = True

    def reset(self):
        """
        Reset the simulator back to its initial conditions and clear the history.
        """
        self.simulator = copy.deepcopy(self.original_simulator)
        self.partner = copy.deepcopy(self.original_partner)
        self.history = {
            "your_choices": [],
            "your_rewards": [],
            "partner_choices": [],
            "partner_rewards": [],
        }
        self.done_simulation = False
        self.total_trials = 0


class HierarchicalParameterRecovery:
    def __init__(
        self,
        estimator: HierarchicalEstimator,
        simulators: Collection[BaseSimulator | ActionSoftmaxSimulator],
        partners: Collection[BaseSimulator],
        bandit_instance: Bandit,
    ) -> None:
        check_params_type({estimator: HierarchicalEstimator, bandit_instance: Bandit})
        if len(simulators) != len(partners):
            raise ValueError(f"The length of `simulators` and `partners` must match.")

        for simulator, partner in zip(simulators, partners):
            check_params_type(
                {
                    simulator: [BaseSimulator, ActionSoftmaxSimulator],
                    partner: BaseSimulator,
                }
            )

        self.estimator = estimator
        self.generators = []
        for simulator, partner in zip(simulators, partners):
            self.generators += [Generator(simulator, partner, bandit_instance)]

        self.history = {
            "your_choices": [],
            "your_rewards": [],
            "partner_choices": [],
            "partner_rewards": [],
            "groups": [],
        }
        self.done_simulation = False

    def simulate(self, n_trials: int, n_sessions) -> None:
        for ind, generator in enumerate(self.generators):
            for _ in range(n_sessions):
                self.history["groups"].append(ind)
                generator.simulate(n_trials)
                for key in [
                    "your_choices",
                    "your_rewards",
                    "partner_choices",
                    "partner_rewards",
                ]:
                    self.history[key].append(generator.history[key])

                generator.reset()

        self.done_simulation = True

    def fit(self, **kwargs):
        if not self.done_simulation:
            raise Exception(
                "You should call simulate first to generate simulation data"
            )

        num_choices = len(self.generators[0].bandit_instance.arms)

        return self.estimator.fit(num_choices, **self.history)

    def reset(self):
        for generator in self.generators:
            generator.reset()

        self.history = {
            "your_choices": [],
            "your_rewards": [],
            "partner_choices": [],
            "partner_rewards": [],
            "groups": [],
        }
        self.done_simulation = False
