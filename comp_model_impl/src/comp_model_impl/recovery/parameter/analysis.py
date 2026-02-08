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


def _compute_error_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    n = int(len(x))
    if n == 0:
        return {"n": 0}

    err = y - x
    abs_err = np.abs(err)

    corr = float(np.corrcoef(x, y)[0, 1]) if n >= 2 else np.nan
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(abs_err))
    med_abs = float(np.median(abs_err))
    bias = float(np.mean(err))

    return {
        "n": n,
        "corr": corr,
        "rmse": rmse,
        "bias": bias,
        "mae": mae,
        "median_abs_error": med_abs,
    }


def compute_population_recovery_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute recovery metrics for population-level quantities.

    Population-level records typically have one estimate per parameter per
    replication. Metrics are therefore pooled across replications (grouped by
    ``param`` only) rather than computed within each replication. A sentinel
    ``rep`` value of ``-1`` is included for schema compatibility.

    Parameters
    ----------
    df : pandas.DataFrame
        Tidy records table with at least ``rep``, ``param``, ``true``, and ``hat`` columns.
        Population tables typically keep the same schema as subject-level tables,
        with ``subject_id`` set to a sentinel value (e.g., ``"POP"``).

    Returns
    -------
    pandas.DataFrame
        Per-parameter metrics pooled across replications.

    Examples
    --------
    >>> import pandas as pd
    >>> pop_df = pd.DataFrame(
    ...     {
    ...         "rep": [0, 1],
    ...         "subject_id": ["POP", "POP"],
    ...         "param": ["alpha", "alpha"],
    ...         "true": [0.2, 0.2],
    ...         "hat": [0.25, 0.3],
    ...     }
    ... )
    >>> metrics = compute_population_recovery_metrics(pop_df)
    >>> list(metrics["param"]) == ["alpha"]
    True
    """
    _require_columns(df, ["rep", "param", "true", "hat"])
    out_rows: list[dict[str, Any]] = []

    for param, g in df.groupby("param", sort=True):
        g2 = g.dropna(subset=["true", "hat"])
        if len(g2) == 0:
            out_rows.append({"param": param, "rep": -1, "n": 0})
            continue

        x = g2["true"].to_numpy(dtype=float)
        y = g2["hat"].to_numpy(dtype=float)
        metrics = _compute_error_metrics(x, y)
        out_rows.append({"param": param, "rep": -1, **metrics})

    return pd.DataFrame(out_rows).sort_values(["param"]).reset_index(drop=True)


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
        if len(g2) == 0:
            out_rows.append({"param": param, "rep": int(rep), "n": 0})
            continue

        x = g2["true"].to_numpy(dtype=float)
        y = g2["hat"].to_numpy(dtype=float)
        metrics = _compute_error_metrics(x, y)
        out_rows.append({"param": param, "rep": int(rep), **metrics})

    return pd.DataFrame(out_rows).sort_values(["param", "rep"]).reset_index(drop=True)
