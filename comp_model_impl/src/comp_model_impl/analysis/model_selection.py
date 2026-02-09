"""Model-selection utilities for fitted model tables."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_information_criteria(
    fit_table: pd.DataFrame,
    *,
    ll_col: str = "ll_total",
    k_col: str = "k_total",
    n_obs_col: str = "n_obs_total",
    waic_col: str | None = None,
    elpd_waic_col: str | None = None,
    group_cols: tuple[str, ...] = ("rep", "generating_model"),
) -> pd.DataFrame:
    """Add model-comparison criteria columns to a fit table.

    Parameters
    ----------
    fit_table : pandas.DataFrame
        Table with one row per fitted model. Must include log-likelihood,
        free-parameter count, and observation count columns.
    ll_col : str, optional
        Column name for total log-likelihood.
    k_col : str, optional
        Column name for total free-parameter count.
    n_obs_col : str, optional
        Column name for observation count.
    waic_col : str or None, optional
        Column name for WAIC values. If ``None``, this function auto-detects a
        ``"waic"`` column when present.
    elpd_waic_col : str or None, optional
        Column name for ELPD-WAIC values. If provided (or auto-detected as
        ``"elpd_waic"``), WAIC is computed as ``-2 * elpd_waic``.
    group_cols : tuple[str, ...], optional
        Columns that define independent model-comparison groups. Metrics such as
        delta criteria and weights are computed within each group. If no
        provided group columns are present, all rows are treated as one group.

    Returns
    -------
    pandas.DataFrame
        Copy of ``fit_table`` with added columns:
        ``aic``, ``bic``, ``delta_aic``, ``delta_bic``,
        ``akaike_weight``, ``bic_weight``,
        ``bf_best_vs_model_bic``, ``bf_model_vs_best_bic``, and when available
        WAIC columns: ``waic``, ``delta_waic``, ``waic_weight``.

    Notes
    -----
    Bayes factors are the common BIC approximation:
    ``BF(best vs model_i) ≈ exp((BIC_i - BIC_best) / 2)``.
    """
    out = fit_table.copy()
    if out.empty:
        for c in (
            "aic",
            "bic",
            "delta_aic",
            "delta_bic",
            "akaike_weight",
            "bic_weight",
            "bf_best_vs_model_bic",
            "bf_model_vs_best_bic",
            "waic",
            "delta_waic",
            "waic_weight",
        ):
            out[c] = np.array([], dtype=float)
        return out

    required = (ll_col, k_col, n_obs_col)
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"fit_table is missing required column(s): {missing}")

    ll = out[ll_col].astype(float)
    k = out[k_col].astype(float)
    n_obs = out[n_obs_col].astype(float).clip(lower=1.0)

    out["aic"] = 2.0 * k - 2.0 * ll
    out["bic"] = k * np.log(n_obs) - 2.0 * ll

    active_group_cols = tuple(c for c in group_cols if c in out.columns)
    if active_group_cols:
        min_aic = out.groupby(list(active_group_cols), dropna=False)["aic"].transform("min")
        min_bic = out.groupby(list(active_group_cols), dropna=False)["bic"].transform("min")
    else:
        min_aic = pd.Series(float(out["aic"].min()), index=out.index)
        min_bic = pd.Series(float(out["bic"].min()), index=out.index)

    out["delta_aic"] = out["aic"] - min_aic
    out["delta_bic"] = out["bic"] - min_bic

    # Relative weights: exp(-0.5 * delta). Clamp exponent for numerical safety.
    out["_aic_rel"] = np.exp(-0.5 * np.clip(out["delta_aic"].astype(float), 0.0, 700.0))
    out["_bic_rel"] = np.exp(-0.5 * np.clip(out["delta_bic"].astype(float), 0.0, 700.0))

    if active_group_cols:
        aic_den = out.groupby(list(active_group_cols), dropna=False)["_aic_rel"].transform("sum")
        bic_den = out.groupby(list(active_group_cols), dropna=False)["_bic_rel"].transform("sum")
    else:
        aic_den = pd.Series(float(out["_aic_rel"].sum()), index=out.index)
        bic_den = pd.Series(float(out["_bic_rel"].sum()), index=out.index)

    out["akaike_weight"] = out["_aic_rel"] / aic_den.replace(0.0, np.nan)
    out["bic_weight"] = out["_bic_rel"] / bic_den.replace(0.0, np.nan)

    out["bf_best_vs_model_bic"] = np.exp(0.5 * np.clip(out["delta_bic"].astype(float), 0.0, 700.0))
    out["bf_model_vs_best_bic"] = np.exp(-0.5 * np.clip(out["delta_bic"].astype(float), 0.0, 700.0))

    waic_source_col = waic_col
    if waic_source_col is None and "waic" in out.columns:
        waic_source_col = "waic"

    elpd_source_col = elpd_waic_col
    if waic_source_col is None and elpd_source_col is None and "elpd_waic" in out.columns:
        elpd_source_col = "elpd_waic"

    if waic_source_col is not None:
        if waic_source_col not in out.columns:
            raise ValueError(f"fit_table is missing required WAIC column: {waic_source_col!r}")
        waic_values = out[waic_source_col].astype(float)
        out["waic"] = waic_values
    elif elpd_source_col is not None:
        if elpd_source_col not in out.columns:
            raise ValueError(f"fit_table is missing required ELPD-WAIC column: {elpd_source_col!r}")
        waic_values = -2.0 * out[elpd_source_col].astype(float)
        out["waic"] = waic_values

    if "waic" in out.columns:
        if active_group_cols:
            min_waic = out.groupby(list(active_group_cols), dropna=False)["waic"].transform("min")
        else:
            min_waic = pd.Series(float(out["waic"].min()), index=out.index)

        out["delta_waic"] = out["waic"] - min_waic
        out["_waic_rel"] = np.exp(-0.5 * np.clip(out["delta_waic"].astype(float), 0.0, 700.0))
        if active_group_cols:
            waic_den = out.groupby(list(active_group_cols), dropna=False)["_waic_rel"].transform("sum")
        else:
            waic_den = pd.Series(float(out["_waic_rel"].sum()), index=out.index)
        out["waic_weight"] = out["_waic_rel"] / waic_den.replace(0.0, np.nan)

    drop_cols = [c for c in ("_aic_rel", "_bic_rel", "_waic_rel") if c in out.columns]
    out.drop(columns=drop_cols, inplace=True)
    return out
