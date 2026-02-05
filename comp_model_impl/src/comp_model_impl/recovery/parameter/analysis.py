"""Analysis utilities for parameter recovery outputs.

This module computes summary statistics from the tidy recovery records table
``[rep, subject_id, param, true, hat]``. It avoids regression-based summaries
and focuses on error- and agreement-based metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def compute_population_recovery_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute recovery metrics for population-level quantities.

    This is a semantic alias of :func:`compute_parameter_recovery_metrics`. It
    exists for readability when you pass a population-level records table.

    Parameters
    ----------
    df : pandas.DataFrame
        Tidy records table with at least ``rep``, ``param``, ``true``, and ``hat`` columns.
        Population tables typically keep the same schema as subject-level tables,
        with ``subject_id`` set to a sentinel value (e.g., ``"POP"``).

    Returns
    -------
    pandas.DataFrame
        Per-parameter/per-rep metrics table (see :func:`compute_parameter_recovery_metrics`).

    Examples
    --------
    >>> import pandas as pd
    >>> pop_df = pd.DataFrame(
    ...     {
    ...         "rep": [0, 0],
    ...         "subject_id": ["POP", "POP"],
    ...         "param": ["alpha", "beta"],
    ...         "true": [0.2, 3.0],
    ...         "hat": [0.25, 2.8],
    ...     }
    ... )
    >>> metrics = compute_population_recovery_metrics(pop_df)
    >>> set(metrics["param"]) == {"alpha", "beta"}
    True
    """

    return compute_parameter_recovery_metrics(df)


def compute_parameter_recovery_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-parameter recovery metrics per replication from a tidy records table.

    The input is expected to contain ``rep``, ``param``, ``true``, and ``hat``
    columns (``subject_id`` and other columns are ignored).

    Metrics computed (per ``param`` × ``rep``):
      - n (count)
      - corr (Pearson correlation between true and hat)
      - rmse (root mean squared error)
      - mae (mean absolute error)
      - median_abs_error
      - bias (mean error = hat - true)
    """
    _require_columns(df, ["rep", "param", "true", "hat"])
    out_rows: list[dict[str, Any]] = []

    for (param, rep), g in df.groupby(["param", "rep"], sort=True):
        g2 = g.dropna(subset=["true", "hat"])
        n = int(len(g2))
        if n == 0:
            out_rows.append({"param": param, "rep": int(rep), "n": 0})
            continue

        x = g2["true"].to_numpy(dtype=float)
        y = g2["hat"].to_numpy(dtype=float)

        err = y - x
        abs_err = np.abs(err)

        corr = float(np.corrcoef(x, y)[0, 1]) if n >= 2 else np.nan
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(abs_err))
        med_abs = float(np.median(abs_err))
        bias = float(np.mean(err))

        out_rows.append(
            {
                "param": param,
                "rep": int(rep),
                "n": n,
                "corr": corr,
                "rmse": rmse,
                "bias": bias,
                "mae": mae,
                "median_abs_error": med_abs,
            }
        )

    return pd.DataFrame(out_rows).sort_values(["param", "rep"]).reset_index(drop=True)
