import numpy as np
from numpy import ndarray
from .base import BaseSimulator
from scipy.special import softmax


class QSoftmaxSimulator(BaseSimulator):
    """
    Implements a Q-learning simulator with a softmax action selection strategy.

    Attributes
    ----------
    lr : float
        The learning rate used to update Q-values.
    beta : float
        A temperature parameter for the softmax function to control exploration vs. exploitation.
    q_values : ndarray
        A numpy array storing Q-values for each action.
    """

    def __init__(self, lr: float, beta: float, initial_values: ndarray) -> None:
        """
        Initialize the QSoftmaxSimulator with learning rate, beta parameter, and initial Q-values.

        Parameters
        ----------
        lr : float
            Learning rate.
        beta : float
            Temperature parameter for the softmax function.
        initial_values : ndarray
            Initial Q-values for each action.
        """
        super().__init__()
        self.lr = lr
        self.beta = beta
        self.q_values = np.array(initial_values)

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
