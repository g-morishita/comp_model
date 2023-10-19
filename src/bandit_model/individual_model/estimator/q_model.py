import numpy as np
from scipy.optimize import LinearConstraint
from scipy.special import softmax

from .base import MLEstimator


class QSotfmaxMLE(MLEstimator):
    def __init__(self) -> None:
        """
        This class estimates free parameters of  a simple Q learning model, which makes a choice using softmax function using the maximum likelihood estimator (MLE).
        The free parameters are a learning rate `lr` and an inverse temperature `beta`.
        """
        super().__init__()

    def neg_ll(self, args):
        lr, beta = args

        # Create a matrix that will indicate which choices were taken at each time
        action_taken = np.zeros((len(self.choices), self.num_choices))
        action_taken[np.arange(len(self.choices)), self.choices] = 1

        # Calculate delta, the prediction error for each choice
        deltas = lr * (
            self.rewards[:, None]
            - np.where(
                action_taken,
                np.roll(action_taken, shift=1, axis=0) @ self.choices[:, None],
                0,
            )
        )

        # Update q-values
        q_vals = np.cumsum(deltas * action_taken, axis=0)

        # Calculate choice probabilities
        choice_prob = softmax(beta * q_vals, axis=1)

        # Negative log-likelihood calculation
        nll = -np.log(
            choice_prob[np.arange(len(self.choices)), self.choices] + 1e-8
        ).sum()

        return nll

    def initialize_params(self) -> np.ndarray:
        init_lr = np.random.beta(2, 2)
        init_beta = np.random.gamma(2, 0.333)
        return np.array([init_lr, init_beta])

    def constraints(self):
        A = np.eye(2)
        lb = np.array([0, 0])
        ub = [1, np.inf]
        return LinearConstraint(A, lb, ub)
