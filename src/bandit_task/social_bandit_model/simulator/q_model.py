import numpy as np
from scipy.special import softmax
from .base import BaseSimulator


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


class ForgetfulQSoftmaxSimulator(QSoftmaxSimulator):
    def __init__(
        self,
        lr_own,
        lr_partner,
        beta,
        forgetfulness_own,
        forgetful_partner,
        initial_values,
    ):
        self.lr_own = lr_own
        self.lr_partner = lr_partner
        self.beta = beta
        self.forgetfulness_own = forgetfulness_own
        self.forgetfulness_partner = forgetful_partner
        self.initial_values = np.array(initial_values, dtype=float)
        self.q_values = np.array(initial_values, dtype=float)

    def learn_from_own(self, choice: int, reward: float) -> None:
        super().learn_from_own(choice, reward)
        for action in range(len(self.q_values)):
            if action != choice:
                self.q_values[action] = (
                    self.forgetfulness_own * self.initial_values[action]
                    + (1 - self.forgetfulness_own) * self.q_values[action]
                )

    def learn_from_partner(self, choice: int, reward: float) -> None:
        super().learn_from_partner(choice, reward)
        for action in range(len(self.q_values)):
            if action != choice:
                self.q_values[action] = (
                    self.forgetfulness_partner * self.initial_values[action]
                    + (1 - self.forgetfulness_partner) * self.q_values[action]
                )


class QSoftmaxInfoBonusSimulator(QSoftmaxSimulator):
    def __init__(self, lr_own, lr_partner, beta, coef_info_bonus, initial_values):
        super().__init__(lr_own, lr_partner, beta, initial_values)
        self.n_chosen = np.ones(len(initial_values))
        self.coef_info_bonus = coef_info_bonus

    def make_choice(self) -> int:
        # Calculate the probability of each action using the softmax function.
        values = self.beta * (
            self.q_values + 1 / np.sqrt(self.n_chosen) * self.coef_info_bonus
        )
        choice_prob = softmax(values)
        # Randomly select an action based on its probability.
        return np.random.choice(len(self.q_values), p=choice_prob)

    def learn_from_own(self, choice: int, reward: float) -> None:
        super().learn_from_own(choice, reward)
        self.n_chosen[choice] += 1

    def learn_from_partner(self, choice: int, reward: float) -> None:
        super().learn_from_partner(choice, reward)
        self.n_chosen[choice] += 1


class QSoftmaxDecayingInfoBonusSimulator(QSoftmaxSimulator):
    def __init__(
        self,
        lr_own,
        lr_partner,
        beta,
        initial_info_bonuses,
        info_decay_rate_own,
        info_decay_rate_partner,
        initial_values,
    ):
        super().__init__(lr_own, lr_partner, beta, initial_values)
        self.n_chosen = np.ones(len(initial_values))
        self.info_bonuses = np.ones(len(initial_values)) * initial_info_bonuses
        self.info_decay_rate_own = info_decay_rate_own
        self.info_decay_rate_partner = info_decay_rate_partner

    def make_choice(self) -> int:
        # Calculate the probability of each action using the softmax function.
        values = self.beta * (self.q_values + self.info_bonuses)
        choice_prob = softmax(values)
        # Randomly select an action based on its probability.
        return np.random.choice(len(self.q_values), p=choice_prob)

    def learn_from_own(self, choice: int, reward: float) -> None:
        super().learn_from_own(choice, reward)
        self.info_bonuses[choice] *= self.info_decay_rate_own

    def learn_from_partner(self, choice: int, reward: float) -> None:
        super().learn_from_partner(choice, reward)
        self.info_bonuses[choice] *= self.info_decay_rate_partner


class StickyQSoftmaxSimulator(QSoftmaxSimulator):
    def __init__(
        self,
        lr_own,
        lr_partner,
        beta,
        stickiness_own,
        stickiness_partner,
        initial_values,
    ):
        super().__init__(lr_own, lr_partner, beta, initial_values)
        self.stickiness_own = stickiness_own
        self.stickiness_partner = stickiness_partner
        self.previous_own_choice = None
        self.previous_partner_choice = None

    def make_choice(self) -> int:
        # Calculate the probability of each action using the softmax function.
        values = self.q_values.copy()
        if self.previous_own_choice is not None:
            values[self.previous_own_choice] += self.stickiness_own
        if self.previous_partner_choice is not None:
            values[self.previous_partner_choice] += self.stickiness_partner

        choice_prob = softmax(values * self.beta)
        # Randomly select an action based on its probability.
        return np.random.choice(len(self.q_values), p=choice_prob)

    def learn_from_own(self, choice: int, reward: float) -> None:
        self.previous_own_choice = choice
        super().learn_from_own(choice, reward)

    def learn_from_partner(self, choice: int, reward: float) -> None:
        self.previous_partner_choice = choice
        super().learn_from_partner(choice, reward)


class StickyQSoftmaxBonusSimulator(QSoftmaxSimulator):
    def __init__(
        self,
        lr_own,
        lr_partner,
        beta,
        coef_info_bonus,
        stickiness_own,
        stickiness_partner,
        initial_values,
    ):
        super().__init__(lr_own, lr_partner, beta, initial_values)
        self.n_chosen = np.ones(len(initial_values))
        self.coef_info_bonus = coef_info_bonus
        self.stickiness_own = stickiness_own
        self.stickiness_partner = stickiness_partner
        self.previous_own_choice = None
        self.previous_partner_choice = None

    def make_choice(self) -> int:
        # add the information bonus
        values = self.q_values + self.coef_info_bonus / np.sqrt(self.n_chosen)

        # add stickiness
        if self.previous_own_choice is not None:
            values[self.previous_own_choice] += self.stickiness_own
        if self.previous_partner_choice is not None:
            values[self.previous_partner_choice] += self.stickiness_partner

        choice_prob = softmax(self.beta * values)
        # Randomly select an action based on its probability.
        return np.random.choice(len(self.q_values), p=choice_prob)

    def learn_from_own(self, choice: int, reward: float) -> None:
        self.previous_own_choice = choice
        super().learn_from_own(choice, reward)
        self.n_chosen[choice] += 1

    def learn_from_partner(self, choice: int, reward: float) -> None:
        self.previous_partner_choice = choice
        super().learn_from_partner(choice, reward)
        self.n_chosen[choice] += 1


class ForgetfulQSoftmaxBonusSimulator(QSoftmaxSimulator):
    def __init__(
        self,
        lr_own,
        lr_partner,
        beta,
        coef_info_bonus,
        f_own,
        f_partner,
        initial_values,
    ):
        super().__init__(lr_own, lr_partner, beta, initial_values)
        self.n_chosen = np.ones(len(initial_values))
        self.coef_info_bonus = coef_info_bonus
        self.f_own = f_own
        self.f_partner = f_partner
        self.initial_values = initial_values

    def make_choice(self) -> int:
        # add the information bonus
        values = self.q_values + 1 / np.sqrt(self.n_chosen) * self.coef_info_bonus
        choice_prob = softmax(self.beta * values)

        # Randomly select an action based on its probability.
        return np.random.choice(len(self.q_values), p=choice_prob)

    def learn_from_own(self, choice: int, reward: float) -> None:
        super().learn_from_own(choice, reward)
        self.n_chosen[choice] += 1

        # Forget
        for action in range(len(self.q_values)):
            if action != choice:
                self.q_values[action] = (
                    self.f_own * self.initial_values[action]
                    + (1 - self.f_own) * self.q_values[action]
                )

    def learn_from_partner(self, choice: int, reward: float) -> None:
        self.previous_partner_choice = choice
        super().learn_from_partner(choice, reward)
        self.n_chosen[choice] += 1

        # Forget
        for action in range(len(self.q_values)):
            if action != choice:
                self.q_values[action] = (
                    self.f_partner * self.initial_values[action]
                    + (1 - self.f_partner) * self.q_values[action]
                )


class StickyForgetfulQSoftmaxSimulator(QSoftmaxSimulator):
    def __init__(
        self,
        lr_own,
        lr_partner,
        beta,
        forgetful_own,
        forgetful_partner,
        stickiness_own,
        stickiness_partner,
        initial_values,
    ):
        super().__init__(lr_own, lr_partner, beta, initial_values)
        self.stickiness_own = stickiness_own
        self.stickiness_partner = stickiness_partner
        self.forgetful_own = forgetful_own
        self.forgetful_partner = forgetful_partner
        self.initial_values = initial_values
        self.previous_own_choice = None
        self.previous_partner_choice = None

    def make_choice(self) -> int:
        # add stickiness
        values = self.q_values.copy()
        if self.previous_own_choice is not None:
            values[self.previous_own_choice] += self.stickiness_own
        if self.previous_partner_choice is not None:
            values[self.previous_partner_choice] += self.stickiness_partner

        choice_prob = softmax(self.beta * values)
        # Randomly select an action based on its probability.
        return np.random.choice(len(self.q_values), p=choice_prob)

    def learn_from_own(self, choice: int, reward: float) -> None:
        self.previous_own_choice = choice
        super().learn_from_own(choice, reward)
        # Forget
        for action in range(len(self.q_values)):
            if action != choice:
                self.q_values[action] = (
                    self.forgetful_own * self.initial_values[action]
                    + (1 - self.forgetful_own) * self.q_values[action]
                )

    def learn_from_partner(self, choice: int, reward: float) -> None:
        self.previous_partner_choice = choice
        super().learn_from_partner(choice, reward)
        # Forget
        for action in range(len(self.q_values)):
            if action != choice:
                self.q_values[action] = (
                    self.forgetful_partner * self.initial_values[action]
                    + (1 - self.forgetful_partner) * self.q_values[action]
                )


class StickyForgetfulQSoftmaxBonusSimulator(QSoftmaxSimulator):
    def __init__(
        self,
        lr_own,
        lr_partner,
        beta,
        coef_info_bonus,
        forgetful_own,
        forgetful_partner,
        stickiness_own,
        stickiness_partner,
        initial_values,
    ):
        super().__init__(lr_own, lr_partner, beta, initial_values)
        self.n_chosen = np.ones(len(initial_values))
        self.coef_info_bonus = coef_info_bonus
        self.stickiness_own = stickiness_own
        self.stickiness_partner = stickiness_partner
        self.forgetful_own = forgetful_own
        self.forgetful_partner = forgetful_partner
        self.initial_values = initial_values
        self.previous_own_choice = None
        self.previous_partner_choice = None

    def make_choice(self) -> int:
        # add the information bonus
        values = self.q_values + self.coef_info_bonus / np.sqrt(self.n_chosen)

        # add stickiness
        if self.previous_own_choice is not None:
            values[self.previous_own_choice] += self.stickiness_own
        if self.previous_partner_choice is not None:
            values[self.previous_partner_choice] += self.stickiness_partner

        choice_prob = softmax(self.beta * values)
        # Randomly select an action based on its probability.
        return np.random.choice(len(self.q_values), p=choice_prob)

    def learn_from_own(self, choice: int, reward: float) -> None:
        self.previous_own_choice = choice
        super().learn_from_own(choice, reward)
        self.n_chosen[choice] += 1
        # Forget
        for action in range(len(self.q_values)):
            if action != choice:
                self.q_values[action] = (
                    self.forgetful_own * self.initial_values[action]
                    + (1 - self.forgetful_own) * self.q_values[action]
                )

    def learn_from_partner(self, choice: int, reward: float) -> None:
        self.previous_partner_choice = choice
        super().learn_from_partner(choice, reward)
        self.n_chosen[choice] += 1
        # Forget
        for action in range(len(self.q_values)):
            if action != choice:
                self.q_values[action] = (
                    self.forgetful_partner * self.initial_values[action]
                    + (1 - self.forgetful_partner) * self.q_values[action]
                )
