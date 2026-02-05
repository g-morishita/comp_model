import numpy as np


def perseveration_bonus(last_choice: int | None, n_actions: int, kappa: float) -> np.ndarray:
    """Return a perseveration bonus vector for the last private choice.

    Parameters
    ----------
    last_choice : int or None
        Last chosen action index (None if no previous choice).
    n_actions : int
        Number of available actions.
    kappa : float
        Perseveration strength (added to the last chosen action).

    Returns
    -------
    numpy.ndarray
        Length-``n_actions`` vector with ``+kappa`` on the last choice and 0
        elsewhere.

    Examples
    --------
    >>> _perseveration_bonus(last_choice=1, n_actions=3, kappa=0.5).tolist()
    [0.0, 0.5, 0.0]
    """
    if last_choice is None or kappa == 0.0:
        return np.zeros(n_actions, dtype=float)
    b = np.zeros(n_actions, dtype=float)
    if 0 <= last_choice < n_actions:
        b[last_choice] = float(kappa)
    return b