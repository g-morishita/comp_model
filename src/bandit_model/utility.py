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
