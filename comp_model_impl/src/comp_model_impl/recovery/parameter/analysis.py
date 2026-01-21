from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_parameter_recovery_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects a tidy table with columns: [rep, subject_id, param, true, hat]

    Returns per-parameter metrics:
      - n
      - corr(true, hat)
      - rmse
      - bias (mean(hat-true))
      - slope/intercept from hat ~ intercept + slope*true
    """
    out_rows: list[dict[str, Any]] = []

    for param, g in df.groupby("param", sort=True):
        g2 = g.dropna(subset=["true", "hat"])
        n = int(len(g2))
        if n == 0:
            out_rows.append({"param": param, "n": 0})
            continue

        x = g2["true"].to_numpy(dtype=float)
        y = g2["hat"].to_numpy(dtype=float)

        corr = float(np.corrcoef(x, y)[0, 1]) if n >= 2 else np.nan
        rmse = float(np.sqrt(np.mean((y - x) ** 2)))
        bias = float(np.mean(y - x))

        if n >= 2 and np.std(x) > 0:
            slope, intercept = np.polyfit(x, y, deg=1)
            slope = float(slope)
            intercept = float(intercept)
        else:
            slope = np.nan
            intercept = np.nan

        out_rows.append(
            {
                "param": param,
                "n": n,
                "corr": corr,
                "rmse": rmse,
                "bias": bias,
                "slope": slope,
                "intercept": intercept,
            }
        )

    return pd.DataFrame(out_rows).sort_values("param").reset_index(drop=True)
