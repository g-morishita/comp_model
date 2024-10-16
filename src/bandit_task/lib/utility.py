import warnings

import numpy as np
from scipy.optimize import minimize


def is_iterable(obj):
    """Check if a variable is interable or not"""
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def check_params_type(param_type_pairs: dict) -> None:
    """
    Check if parameters are compatible with given types.

    Parameters
    ----------
    param_type_pairs : dict
        A dictionary to keep pairs of a parameter and its expected type.

    Returns
    -------
    """
    for param, type in param_type_pairs.items():
        if not is_iterable(type):
            type = [type]

        match = False
        for t in type:
            if isinstance(param, t):
                match = True

        if not match:
            raise ValueError(
                f"{param} should be inherited from {type} classes. {param.__class__.__name__} is given."
            )


def read_options(allowed_keywords: set, **kwargs: dict) -> dict:
    """
    Extracts keyword arguments that match the set of allowed keywords.

    The function filters out keyword arguments that are not present in
    the `allowed_keywords` set. This is useful for cases where you want to
    pass a large number of keyword arguments to a function but only process
    a subset of them.

    Parameters
    ----------
    allowed_keywords : set
        A set containing the names of keyword arguments that should be
        extracted and returned.

    kwargs : dict
        Arbitrary keyword arguments that will be filtered based on
        their presence in the `allowed_keywords` set.

    Returns
    -------
    dict
        A dictionary containing the filtered keyword arguments that are
        present in the `allowed_keywords` set.
    """

    options = {}
    for k, v in kwargs.items():
        # Check if the keyword is allowed and add to the options dictionary
        if k in allowed_keywords:
            options[k] = v

    return options


def optimize_non_convex_obj(obj, init_param, constraints, method, n_trials, options):
    # Initialize variables for optimization results
    min_nll = np.inf  # Minimum negative log-likelihood
    opt_param = None  # Optimal parameters

    # Optimize using multiple initializations
    for _ in range(n_trials):
        result = minimize(
            obj,
            init_param,
            method=method,
            constraints=constraints,
            options=options,
        )

        # Warning if optimization was not successful
        if not result.success:
            warnings.warn(result.message)
        else:
            # Update the best parameters if new result is better
            if min_nll > result.fun:
                min_nll = result.fun
                opt_param = result.x

    if opt_param is None:
        warnings.warn("The estimation did not work. An empty array is returned.")
        return np.nan, np.array([])

    return min_nll, opt_param
