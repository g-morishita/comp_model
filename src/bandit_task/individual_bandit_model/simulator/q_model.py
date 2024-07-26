import numpy as np
from numpy import ndarray
from .base import BaseSimulator
from scipy.special import softmax
from typing import Sequence


class QSoftmaxSimulator(BaseSimulator):
    """
    Implements a Q-learning simulator with a softmax action selection strategy.

    Attributes
    ----------
    lr : float
        The learning rate used to update Q-values.
    beta : float
        A temperature parameter for the softmax function to control exploration vs. exploitation.
    q_values : Sequence[int, float]
        A numpy array storing Q-values for each action.
    """

    def __init__(
        self, lr: float, beta: float, initial_values: Sequence[int | float]
    ) -> None:
        """
        Initialize the QSoftmaxSimulator with learning rate, beta parameter, and initial Q-values.

        Parameters
        ----------
        lr : float
            Learning rate.
        beta : float
            Temperature parameter for the softmax function.
        initial_values : Sequence[int, float]
            Initial Q-values for each action.
        """
        super().__init__()
        self.lr = lr
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

    def learn(self, choice: int, reward: float) -> None:
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
        self.q_values[choice] = self.q_values[choice] + self.lr * delta


class ForgetfulQSoftmaxSimulator(QSoftmaxSimulator):
    """
    Implements a forgetful Q-learning simulator with a softmax action selection strategy.
    forgetful Q-learning model has a parameter to decay the values of unchosen options.

    Attributes
    ----------
    lr : float
        The learning rate used to update Q-values.
    beta : float
        A temperature parameter for the softmax function to control exploration vs. exploitation.
    forgetfulness : float
        A forgetful parameter
    initial_values
        initial values

    q_values : Sequence[int, float]
        A numpy array storing Q-values for each action.
    """

    def __init__(self, lr: float, beta: float, forgetfulness: float, initial_values):
        """
        Initialize the QSoftmaxSimulator with learning rate, beta parameter, and initial Q-values.

        Parameters
        ----------
        lr : float
            Learning rate.
        beta : float
            Temperature parameter for the softmax function.
        forgetfulness: float
            A forgetful parameter
        initial_values : Sequence[int, float]
            Initial Q-values for each action.
        """
        super().__init__(lr, beta, initial_values)
        self.initial_values = initial_values
        self.forgetfulness = forgetfulness

    def learn(self, choice: int, reward: float) -> None:
        """
        Update the Q-value for the chosen action based on the received reward.
        Unlike the standard Q learner, decay the values of unchosen options.

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
        self.q_values[choice] = self.q_values[choice] + self.lr * delta

        for action in range(len(self.q_values)):
            if action != choice:
                self.q_values[action] = (
                    self.initial_values[action]
                    + (1 - self.forgetfulness) * self.q_values[action]
                )


class StickyQSoftmaxSimulator(BaseSimulator):
    """
    Implements a sticky Q-learning simulator with a softmax action selection strategy.
    Sticky Q-learning model has a parameter to add some value to a previous action.

    Attributes
    ----------
    lr : float
        The learning rate used to update Q-values.
    beta : float
        A temperature parameter for the softmax function to control exploration vs. exploitation.
    stickiness : float
        A stickiness parameter

    q_values : Sequence[int, float]
        A numpy array storing Q-values for each action.
    """

    def __init__(
        self,
        lr: float,
        beta: float,
        stickiness: float,
        initial_values: Sequence[int | float],
    ) -> None:
        """
        Initialize the QSoftmaxSimulator with learning rate, beta parameter, and initial Q-values.

        Parameters
        ----------
        lr : float
            Learning rate.
        beta : float
            Temperature parameter for the softmax function.
        initial_values : Sequence[int, float]
            Initial Q-values for each action.
        """
        super().__init__()
        self.lr = lr
        self.beta = beta
        self.previous_choice = None
        self.stickiness = stickiness
        self.q_values = np.array(initial_values, dtype=float)

    def make_choice(self) -> int:
        """
        Make a choice (i.e., select an action) based on the values and the stickiness using the softmax policy.

        Returns
        -------
        int
            The index of the selected action.
        """
        # Calculate the probability of each action using the softmax function.
        values = self.q_values.copy()
        if self.previous_choice is not None:
            values[self.previous_choice] += self.stickiness
        choice_prob = softmax(values * self.beta)

        # Randomly select an action based on its probability.
        choice = np.random.choice(len(self.q_values), p=choice_prob)
        # Remember what action is chosen for the stickiness
        self.previous_choice = choice
        return choice

    def learn(self, choice: int, reward: float) -> None:
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
        self.q_values[choice] = self.q_values[choice] + self.lr * delta
