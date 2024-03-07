import numpy as np
from typing import Sequence
from scipy.special import softmax


class RewardActionHybridSimulator:
    """
    Implements a reward and action hybrid learning simulator with a softmax action selection strategy.

    Attributes
    ----------
    lr_for_reward : float
        The learning rate used to update Q values.
    lr_for_action : float
        The learning rate used to update action values.
    beta_for_reward : float
        A temperature parameter for Q-values.
    beta_for_action : float
        A temperature parameter for action values.
    q_values : Sequence[float]
        A numpy array storing reward values for each action.
    action_values : Sequence[float]
        A numpy array storing action values for each action.
    """

    def __init__(self, lr_for_reward: float, lr_for_action: float, beta_for_reward: float, beta_for_action: float, initial_values: Sequence[float]) -> None:
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
        self.lr_for_reward = lr_for_reward
        self.lr_for_action = lr_for_action
        self.beta_for_reward = beta_for_reward
        self.beta_for_action = beta_for_action
        self.q_values = np.array(initial_values)
        self.action_values = np.array(initial_values)

    def make_choice(self) -> int:
        """
        Make a choice (i.e., select an action) based on the action values and the softmax policy.

        Returns
        -------
        int
            The index of the selected action.
        """
        combined_values = self.q_values * self.beta_for_reward + self.action_values * self.beta_for_action
        # Calculate the probability of each action using the softmax function.
        choice_prob = softmax(combined_values)
        # Randomly select an action based on its probability.
        return np.random.choice(len(self.action_values), p=choice_prob)

    def learn(self, partner_choice: int, partner_reward: int) -> None:
        """
        Update the action value for the partner's choice

        Parameters
        ----------
        partner_choice : int
            The index of the partner's choice
        """
        # Update the q values for the partner's choice
        self.q_values[partner_choice] = self.q_values[partner_choice] + self.lr_for_reward * (
                partner_reward - self.q_values[partner_choice]
        )

        # Update the action value for the partner's choice
        self.action_values[partner_choice] = (
                self.action_values[partner_choice] + self.lr_for_action * (1 - self.action_values[partner_choice])
        )

        for unchosen_choice in range(len(self.action_values)):
            if unchosen_choice != partner_choice:
                self.action_values[unchosen_choice] = (
                    self.action_values[unchosen_choice] + self.lr_for_action * (0 - self.action_values[unchosen_choice])
                )


class RewardImmediateActionHybridSimulator:
    """
    Implements a reward and action hybrid learning simulator with a softmax action selection strategy.
    The difference between `RewardActionHybridSimulator` is the way of updating and holding action values.
    This class updates an action value for a choice that is just taken to 1 and others to 0.
    This is equivalent of fixing `lr_for_action` to 1.

    Attributes
    ----------
    lr_for_reward : float
        The learning rate used to update Q values.
    beta_for_reward : float
        A temperature parameter for Q-values.
    beta_for_action : float
        A temperature parameter for action values.
    q_values : Sequence[float]
        A numpy array storing reward values for each action.
    action_values : Sequence[float]
        A numpy array storing action values for each action.
    """

    def __init__(self, lr_for_reward: float, beta_for_reward: float, beta_for_action: float, initial_values: Sequence[float]) -> None:
        """
        Initialize the RewardImmediateActionHybridSimulator with learning rate, beta parameter, and initial action values.

        Parameters
        ----------
        lr_for_reward : float
            The learning rate used to update Q values.
        beta_for_reward : float
            A temperature parameter for Q-values.
        beta_for_action : float
            A temperature parameter for action values.
        initial_values : ndarray
            Initial values for each action.
        """
        super().__init__()
        self.lr_for_reward = lr_for_reward
        self.beta_for_reward = beta_for_reward
        self.beta_for_action = beta_for_action
        self.q_values = np.array(initial_values)
        self.action_values = np.array(initial_values)

    def make_choice(self) -> int:
        """
        Make a choice (i.e., select an action) based on the action values and the softmax policy.

        Returns
        -------
        int
            The index of the selected action.
        """
        combined_values = self.q_values * self.beta_for_reward + self.action_values * self.beta_for_action
        # Calculate the probability of each action using the softmax function.
        choice_prob = softmax(combined_values)
        # Randomly select an action based on its probability.
        return np.random.choice(len(self.action_values), p=choice_prob)

    def learn(self, partner_choice: int, partner_reward: int) -> None:
        """
        Update the action value for the partner's choice

        Parameters
        ----------
        partner_choice : int
            The index of the partner's choice
        partner_reward : int
            a reward the partner obtains
        """
        # Update the q values for the partner's choice
        self.q_values[partner_choice] = self.q_values[partner_choice] + self.lr_for_reward * (
                partner_reward - self.q_values[partner_choice]
        )

        # Update the action value for the partner's choice
        self.action_values[partner_choice] = 1

        for unchosen_choice in range(len(self.action_values)):
            if unchosen_choice != partner_choice:
                self.action_values[unchosen_choice] = 0
