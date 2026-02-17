"""Analysis utilities for model recovery outputs."""

from __future__ import annotations

import pandas as pd


def confusion_matrix(winners: pd.DataFrame) -> pd.DataFrame:
    """Compute a confusion matrix from the winners table.

    Parameters
    ----------
    winners : pandas.DataFrame
        Output table from :func:`run_model_recovery`, containing at least:
        ``generating_model`` and ``selected_model``.

    Returns
    -------
    pandas.DataFrame
        Confusion matrix with generating models as rows and selected models as columns.
    """
    if winners.empty:
        return pd.DataFrame()

    if "generating_model" not in winners or "selected_model" not in winners:
        raise ValueError("winners table must contain generating_model and selected_model columns")

    return pd.crosstab(
        index=winners["generating_model"],
        columns=winners["selected_model"],
        rownames=["generating_model"],
        colnames=["selected_model"],
        dropna=False,
    )
