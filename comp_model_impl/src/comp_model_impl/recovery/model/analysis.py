"""Analysis utilities for model recovery outputs."""

from __future__ import annotations

import numpy as np
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

def summarize_delta_scores(winners: pd.DataFrame) -> pd.DataFrame:
    """Summarize winner-vs-second score gaps by generating model.

    Parameters
    ----------
    winners : pandas.DataFrame
        Winners table expected to include ``generating_model`` and
        ``delta_to_second``.

    Returns
    -------
    pandas.DataFrame
        Summary table with columns ``delta_mean``, ``delta_median``,
        ``delta_p10``, and ``delta_p90`` by generating model.
    """
    if winners.empty or "delta_to_second" not in winners:
        return pd.DataFrame(columns=["generating_model", "delta_mean", "delta_median", "delta_p10", "delta_p90"])

    out = (
        winners.groupby("generating_model")["delta_to_second"]
        .agg(
            delta_mean="mean",
            delta_median="median",
            delta_p10=lambda x: float(np.nanpercentile(x, 10)),
            delta_p90=lambda x: float(np.nanpercentile(x, 90)),
        )
        .reset_index()
    )
    return out
