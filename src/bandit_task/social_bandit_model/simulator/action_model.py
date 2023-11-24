import numpy as np
from numpy import ndarray
from scipy.special import softmax


class ActionSoftmaxSimulator:
    """
    Implements an action learning simulator with a softmax action selection strategy.

    Attributes
    ----------
    lr : float
        The learning rate used to update Q-values.
    beta : float
        A temperature parameter for the softmax function to control exploration vs. exploitation.
    action_values : ndarray
        A numpy array storing values for each action.
    """

    def __init__(self, lr: float, beta: float, initial_values: ndarray) -> None:
        """
        Initialize the ActionSoftmaxSimulator with learning rate, beta parameter, and initial action values.

        Parameters
        ----------
        lr : float
            Learning rate.
        beta : float
            Temperature parameter for the softmax function.
        initial_values : ndarray
            Initial values for each action.
        """
        super().__init__()
        self.lr = lr
        self.beta = beta
        self.action_values = np.array(initial_values) / sum(initial_values)  # normalize action values

    def make_choice(self) -> int:
        """
        Make a choice (i.e., select an action) based on the action values and the softmax policy.

        Returns
        -------
        int
            The index of the selected action.
        """
        # Calculate the probability of each action using the softmax function.
        choice_prob = softmax(self.action_values * self.beta)
        # Randomly select an action based on its probability.
        return np.random.choice(len(self.action_values), p=choice_prob)

    def learn(self, partner_choices: int) -> None:
        """
        Update the action value for the partner's choice

        Parameters
        ----------
        partner_choices : int
            The index of the partner's choice
        """
        delta = 1 - self.action_values[partner_choices]
        # Update the action value for the partner's choice
        self.action_values[partner_choices] = (
            self.action_values[partner_choices] + self.lr * delta
        )

        for unchosen_choice in range(len(self.action_values)):
            if unchosen_choice != partner_choices:
                self.action_values[unchosen_choice] = (
                    self.action_values[unchosen_choice] - self.lr * delta / (len(self.action_values) - 1)
                )
