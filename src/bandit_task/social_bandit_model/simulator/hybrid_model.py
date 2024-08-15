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
    beta : float
        A temperature parameter
    weights_for_value: float
        Weights for q values
    q_values : Sequence[float]
        A numpy array storing reward values for each action.
    action_values : Sequence[float]
        A numpy array storing action values for each action.
    """

    def __init__(
        self,
        lr_for_reward: float,
        lr_for_action: float,
        beta: float,
        weights_for_value: float,
        initial_q_values: Sequence[float],
        initial_action_values: Sequence[float],
    ) -> None:
        """
        Initialize the ActionSoftmaxSimulator with learning rate, beta parameter, and initial action values.

        Parameters
        ----------
        lr_for_reward : float
            The learning rate used to update Q values.
        lr_for_action : float
            The learning rate used to update action values.
        beta : float
            A temperature parameter
        weights_for_value: float
            Weights for q values
        initial_q_values : ndarray
            Initial values for q values.
        initial_action_values : ndarray
            Initial values for action values.
        """
        super().__init__()
        self.lr_for_reward = lr_for_reward
        self.lr_for_action = lr_for_action
        self.beta = beta
        self.weights_for_values = weights_for_value
        self.q_values = np.array(initial_q_values)
        self.action_values = np.array(initial_action_values)

    def make_choice(self) -> int:
        """
        Make a choice (i.e., select an action) based on the action values and the softmax policy.

        Returns
        -------
        int
            The index of the selected action.
        """
        combined_values = (
            self.weights_for_values * self.q_values
            + (1 - self.weights_for_values) * self.action_values
        )
        # Calculate the probability of each action using the softmax function.
        choice_prob = softmax(combined_values * self.beta)
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
        self.q_values[partner_choice] = self.q_values[
            partner_choice
        ] + self.lr_for_reward * (partner_reward - self.q_values[partner_choice])

        # Update the action value for the partner's choice
        self.action_values[partner_choice] = self.action_values[
            partner_choice
        ] + self.lr_for_action * (1 - self.action_values[partner_choice])

        for unchosen_choice in range(len(self.action_values)):
            if unchosen_choice != partner_choice:
                self.action_values[unchosen_choice] = self.action_values[
                    unchosen_choice
                ] + self.lr_for_action * (0 - self.action_values[unchosen_choice])


class ForgetfulRewardActionHybridSimulator:
    """
    Implements a reward and action hybrid learning simulator with a softmax action selection strategy.

    Attributes
    ----------
    lr_for_reward : float
        The learning rate used to update Q values.
    lr_for_action : float
        The learning rate used to update action values.
    forget_rate : float
        The forgetfulness rate
    beta : float
        A temperature parameter
    weights_for_value: float
        Weights for q values
    q_values : Sequence[float]
        A numpy array storing reward values for each action.
    action_values : Sequence[float]
        A numpy array storing action values for each action.
    """

    def __init__(
        self,
        lr_for_reward: float,
        lr_for_action: float,
        forget_rate: float,
        beta: float,
        weights_for_value: float,
        initial_q_values: Sequence[float],
        initial_action_values: Sequence[float],
    ) -> None:
        """
        Initialize the ActionSoftmaxSimulator with learning rate, beta parameter, and initial action values.

        Parameters
        ----------
        lr_for_reward : float
            The learning rate used to update Q values.
        lr_for_action : float
            The learning rate used to update action values.
        forget_rate : float
            The forgetfulness rate
        beta : float
            A temperature parameter
        weights_for_value: float
            Weights for q values
        initial_q_values : ndarray
            Initial values for q values.
        initial_action_values : ndarray
            Initial values for action values.
        """
        super().__init__()
        self.lr_for_reward = lr_for_reward
        self.lr_for_action = lr_for_action
        self.forget_rate = forget_rate
        self.beta = beta
        self.weights_for_values = weights_for_value
        self.initial_q_values = initial_q_values
        self.q_values = np.array(initial_q_values)
        self.action_values = np.array(initial_action_values)

    def make_choice(self) -> int:
        """
        Make a choice (i.e., select an action) based on the action values and the softmax policy.

        Returns
        -------
        int
            The index of the selected action.
        """
        combined_values = (
            self.weights_for_values * self.q_values
            + (1 - self.weights_for_values) * self.action_values
        )
        # Calculate the probability of each action using the softmax function.
        choice_prob = softmax(combined_values * self.beta)
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
        self.q_values[partner_choice] = self.q_values[
            partner_choice
        ] + self.lr_for_reward * (partner_reward - self.q_values[partner_choice])

        # Update the action value for the partner's choice
        self.action_values[partner_choice] = self.action_values[
            partner_choice
        ] + self.lr_for_action * (1 - self.action_values[partner_choice])

        for unchosen_choice in range(len(self.action_values)):
            if unchosen_choice != partner_choice:
                self.action_values[unchosen_choice] = self.action_values[
                    unchosen_choice
                ] + self.lr_for_action * (0 - self.action_values[unchosen_choice])

                self.q_values[unchosen_choice] = (
                    self.initial_q_values[unchosen_choice] * self.forget_rate
                    + (1 - self.forget_rate) * self.q_values[unchosen_choice]
                )


class RewardImmediateActionHybridSimulator:
    """
    Attributes
    ----------
    lr_for_reward : float
        The learning rate used to update Q values.
    beta_for_reward : float
        A temperature parameter for Q-values.
    q_values : Sequence[float]
        A numpy array storing reward values for each action.
    stickiness : float
    """

    def __init__(
        self,
        lr_for_reward: float,
        beta: float,
        stickiness: float,
        initial_values: Sequence[float],
    ) -> None:
        """
        Initialize the RewardImmediateActionHybridSimulator with learning rate, beta parameter, and initial action values.

        Parameters
        ----------
        lr_for_reward : float
            The learning rate used to update Q values.
        beta : float
            A temperature parameter for Q-values.
        initial_values : ndarray
            Initial values for each action.
        """
        super().__init__()
        self.lr_for_reward = lr_for_reward
        self.beta = beta
        self.q_values = np.array(initial_values)
        self.previous_choice = None
        self.stickiness = stickiness

    def make_choice(self) -> int:
        """
        Make a choice (i.e., select an action) based on the action values and the softmax policy.

        Returns
        -------
        int
            The index of the selected action.
        """
        combined_values = self.q_values.copy()
        if self.previous_choice is not None:
            combined_values[self.previous_choice] += self.stickiness
        # Calculate the probability of each action using the softmax function.
        choice_prob = softmax(combined_values * self.beta)
        # Randomly select an action based on its probability.
        return np.random.choice(len(self.q_values), p=choice_prob)

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
        self.q_values[partner_choice] = self.q_values[
            partner_choice
        ] + self.lr_for_reward * (partner_reward - self.q_values[partner_choice])

        # Update the action value for the partner's choice
        self.previous_choice = partner_choice
