import numpy as np
from scipy.special import softmax
from abc import ABC, abstractmethod


class BaseSimulator(ABC):
    @abstractmethod
    def make_choice(self):
        pass

    @abstractmethod
    def learn_from_own(self, choice: int, reward: float) -> None:
        pass

    @abstractmethod
    def learn_from_partner(self, choice: int, reward: float) -> None:
        pass


class QSoftmaxSimulator(BaseSimulator):
    def __init__(self, lr_own, lr_partner, beta, initial_values):
        self.lr_own = lr_own
        self.lr_partner = lr_partner
        self.beta = beta
        self.q_values = np.array(initial_values, dtype=float)

    def make_choice(self) -> int:
        """
        Make a choice (i.e., select an action) based on the Q-values and the softmax policy.

        Returns
        -------
        int
            The index of the selected action.
        """
        # Calculate the probability of each action using the softmax function.
        choice_prob = softmax(self.q_values * self.beta)
        # Randomly select an action based on its probability.
        return np.random.choice(len(self.q_values), p=choice_prob)

    def learn_from_own(self, choice: int, reward: float) -> None:
        """
        Update the Q-value for the chosen action based on the received reward.

        Parameters
        ----------
        choice : int
            The index of the chosen action.
        reward : float
            The received reward after taking the action.
        """
        # Calculate the difference between the received reward and the current Q-value of the action.
        delta = reward - self.q_values[choice]
        # Update the Q-value of the action.
        self.q_values[choice] = self.q_values[choice] + self.lr_own * delta


    def learn_from_partner(self, choice: int, reward: float) -> None:
        """
        Update the Q-value for partner's choice and the received reward.

        Parameters
        ----------
        choice : int
            The index of the chosen action.
        reward : float
            The received reward after taking the action.
        """
        # Calculate the difference between the received reward and the current Q-value of the action.
        delta = reward - self.q_values[choice]
        # Update the Q-value of the action.
        self.q_values[choice] = self.q_values[choice] + self.lr_partner * delta
