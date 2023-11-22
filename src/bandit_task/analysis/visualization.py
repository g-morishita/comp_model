from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def calculate_individual_correct_rates(
    choices: Sequence[int],
    reward_prob: Sequence[float | Sequence],
    correct_choice_rank: int,
    window_size: int,
) -> Sequence[float]:
    """Calculate the correct rates in the course of trials using window size.

    Parameters
    ----------
    choices : int
        A sequence of choices

    reward_prob : Sequence[float, Sequence]
        Reward probabilities of a bandit instance. If the reward probabilities change in the course of trials,
        you can give you a sequence of reward probabilities.

    correct_choice_rank : int
        The rank position of the choice considered as correct, based on its reward probability.
        For instance, a `correct_choice_rank` of 1 means the choice with the highest reward probability is correct,
        while a `correct_choice_rank` of 2 means the second highest, and so on.

    window_size : int
        The number of trials being processed to calculate a correct rate at a time.

    Returns
    -------
        Correct rates

    """
    if correct_choice_rank > len(reward_prob):
        raise ValueError(f"`correct_choice_rank exceeds the number of choices (`len(prob_reward)`)."
                         f"You should specify `correct_choice_rank` less than or euqal to `len(prob_reward)`")
    # If you use np.array, a pd.Series of lists will be converted to np.array of lists not multidimensional np.array.
    # To avoid this, pn.stack is used here.
    reward_prob = np.stack(reward_prob)
    choices = np.array(choices)

    n_trials = len(choices)
    # If prob_reward is one dimension, then extend it to be (the number of trials) dimension.
    # For example, if `reward_prob` is [0.3, 0.2, 0.7] and the number of trials is 3, then
    #  `reward_prob` is going to be:
    # [[0.3, 0.2, 0.7],
    #  [0.3, 0.2, 0.7],
    #  [0.3, 0.2, 0.7]]
    if reward_prob.ndim == 1:
        reward_prob = np.tile(reward_prob, [n_trials, 1])

    # Correct choice is a choice with the `correct_choice_rank`th highest reward probability.
    correct_choices = np.argsort(reward_prob)[:, -correct_choice_rank]
    is_correct = choices == correct_choices
    return is_correct.reshape(-1, window_size).mean(axis=1)


def calculate_average_correct_rates(
    choices: Sequence[Sequence[int]],
    reward_prob: Sequence[float | Sequence[int]],
    correct_choice_rank: int
) -> Sequence[float]:
    """
    Plot the average correct rates over the sessions.

    Parameters
    ----------
    choices : Sequence[Sequence[int]]
        Sequences of choices. The rows correspond to sessions. The columns correspond to trials.

    reward_prob : Sequence[float | Sequence[int]]
        Reward probabilities of a bandit instance. If the reward probabilities change in the course of trials,
        you can give sequences of reward probabilities.
        The rows correspond to sessions. The columns correspond to trials.

    correct_choice_rank : int
        The rank position of the choice considered as correct, based on its reward probability.
        For instance, a `correct_choice_rank` of 1 means the choice with the highest reward probability is correct,
        while a `correct_choice_rank` of 2 means the second highest, and so on.

    Returns
    -------
        Average correct rates

    """
    choices = np.array(choices)
    reward_prob = np.array(reward_prob)
    n_sessions = choices.shape[0]
    n_trials = choices.shape[1]
    n_choices = reward_prob.shape[1]
    if n_sessions != reward_prob.shape[0]:
        raise ValueError(f"`reward_prob.shape[0]` should match the number of sessions.")

    # TODO: add comments
    if reward_prob.ndim == 2:
        reward_prob = reward_prob.reshape(n_sessions, 1, n_choices)
        reward_prob = np.tile(reward_prob, (1, n_trials, 1))

    correct_choices = np.argsort(reward_prob)[:, :, -correct_choice_rank]

    is_correct = correct_choices == choices

    return is_correct.mean(axis=0)


def plot_average_correct_rates(
    choices: Sequence[Sequence[int]],
    reward_prob: Sequence[float | Sequence[int]],
    correct_choice_rank: int,
    path: str
) -> None:
    """
    Plot the average correct rates over the sessions.

    Parameters
    ----------
    choices : Sequence[Sequence[int]]
        Sequences of choices. The rows correspond to sessions. The columns correspond to trials.

    reward_prob : Sequence[float | Sequence[int]]
        Reward probabilities of a bandit instance. If the reward probabilities change in the course of trials,
        you can give sequences of reward probabilities.
        The rows correspond to sessions. The columns correspond to trials.

    correct_choice_rank : int
        The rank position of the choice considered as correct, based on its reward probability.
        For instance, a `correct_choice_rank` of 1 means the choice with the highest reward probability is correct,
        while a `correct_choice_rank` of 2 means the second highest, and so on.

    path : str
        The path to save plots

    Returns
    -------
        Average correct rates

    """
    correct_rates = calculate_average_correct_rates(choices, reward_prob, correct_choice_rank)

    fig, ax1 = plt.subplots()

    ax1.set_title("Average learning curve")
    ax1.set_ylabel(r"$\Pr(Correct)$")
    ax1.set_xlabel("Trials")
    ax1.plot(correct_rates, "blue")
    ax1.set_xticks(np.arange(0, len(correct_rates)))
    ax1.xaxis.grid()
    ax1.yaxis.grid()
    ax1.set_ylim(0, 1.01)
    ax1.set_xlim(0, len(correct_rates) - 1)

    fig.set_size_inches(12, 5)
    fig.set_dpi(100)

    fig.savefig(path, dpi=300, format="png", bbox_inches="tight")


def plot_individual_learning_curve(
    choices: Sequence[int],
    reward_prob: Sequence[float | Sequence],
    correct_choice_rank: int,
    window_size: int,
    path: str,
) -> None:
    """Plot the individual learning curve in the course of trials using window size.

    Parameters
    ----------
    choices : int
        A sequence of choices

    reward_prob : Sequence[float, Sequence]
        Reward probabilities of a bandit instance. If the reward probabilities change in the course of trials,
        you can give you a sequence of reward probabilities.

    window_size : int
        The number of trials being processed to calculate a correct rate at a time.

    correct_choice_rank : int
        The rank position of the choice considered as correct, based on its reward probability.
        For instance, a `correct_choice_rank` of 1 means the choice with the highest reward probability is correct,
        while a `correct_choice_rank` of 2 means the second highest, and so on.

    path : str
        The path to save a plot

    Returns
    -------

    """
    correct_rates = calculate_individual_correct_rates(
        choices, reward_prob, correct_choice_rank, window_size
    )

    fig, ax1 = plt.subplots()

    ax1.set_title("Individual learning curve")
    ax1.set_ylabel(r"$\Pr(Correct)$")
    ax1.set_xlabel("Trials / window_size")
    ax1.plot(correct_rates, "blue")
    ax1.set_xticks(np.arange(0, len(correct_rates)))
    ax1.xaxis.grid()
    ax1.yaxis.grid()
    ax1.set_ylim(0, 1.01)
    ax1.set_xlim(0, len(correct_rates) - 1)

    fig.set_size_inches(7, 5)
    fig.set_dpi(100)

    fig.savefig(path, dpi=300, format="png", bbox_inches="tight")
