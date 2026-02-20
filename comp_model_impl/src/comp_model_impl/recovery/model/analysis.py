"""Analysis utilities for model recovery outputs."""

from __future__ import annotations

import pandas as pd


def confusion_matrix(winners: pd.DataFrame) -> pd.DataFrame:
    """Compute a confusion matrix from the winners table.

    Parameters
    ----------
    winners : pandas.DataFrame
        Output table from :func:`run_model_recovery`, containing at least:
        ``generating_model``, ``selected_model``, and ``winner_determined``.

    Returns
    -------
    pandas.DataFrame
        Confusion matrix with generating models as rows and selected models as columns.
    """
    if winners.empty:
        return pd.DataFrame()

    if (
        "generating_model" not in winners
        or "selected_model" not in winners
        or "winner_determined" not in winners
    ):
        raise ValueError(
            "winners table must contain generating_model, selected_model, and winner_determined columns"
        )

    # Exclude undetermined winner rows.
    w = winners.loc[winners["winner_determined"] == True]  # noqa: E712
    if w.empty:
        return pd.DataFrame()

    return pd.crosstab(
        index=w["generating_model"],
        columns=w["selected_model"],
        rownames=["generating_model"],
        colnames=["selected_model"],
        dropna=False,
    )
