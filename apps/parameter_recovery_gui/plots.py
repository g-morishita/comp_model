"""Parameter recovery plotting focused on cross-parameter color encoding.

This module provides utilities to visualize recovery for a *target* parameter
while coloring points by the value of a *different* parameter. This makes it
easy to inspect regions of one parameter when another parameter is high (e.g.,
learning-rate recovery when inverse temperature is large).

The input is expected to be the long-form recovery records table with columns:
``rep``, ``subject_id``, ``param``, ``true``, and ``hat``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ValueField = Literal["true", "hat", "error", "abs_error"]


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Validate that a DataFrame contains required columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to validate.
    required : Iterable[str]
        Column names that must be present.

    Raises
    ------
    ValueError
        If any required columns are missing.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def _merge_target_and_color(
    *,
    df: pd.DataFrame,
    target_param: str,
    color_param: str,
) -> pd.DataFrame:
    """Join target and color parameter rows into a single table.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-form recovery table.
    target_param : str
        Target parameter name.
    color_param : str
        Color parameter name.

    Returns
    -------
    pandas.DataFrame
        Merged table with target and color values aligned per subject/rep.
    """
    join_cols = [c for c in ("rep", "subject_id") if c in df.columns]
    if not join_cols:
        raise ValueError("DataFrame must include at least one join key: 'rep' or 'subject_id'.")

    target = df.loc[df["param"] == target_param, join_cols + ["true", "hat"]].copy()
    color = df.loc[df["param"] == color_param, join_cols + ["true", "hat"]].copy()
    if target.empty:
        raise ValueError(f"No rows found for target_param={target_param!r}.")
    if color.empty:
        raise ValueError(f"No rows found for color_param={color_param!r}.")

    target.rename(columns={"true": "target_true", "hat": "target_hat"}, inplace=True)
    color.rename(columns={"true": "color_true", "hat": "color_hat"}, inplace=True)

    merged = target.merge(color, on=join_cols, how="inner")
    if merged.empty:
        raise ValueError("No matching rows after merging target and color parameters.")
    return merged


def _extract_value(df: pd.DataFrame, *, prefix: str, field: ValueField) -> np.ndarray:
    """Extract a value field from a merged parameter table.

    Parameters
    ----------
    df : pandas.DataFrame
        Merged table with ``{prefix}_true`` and ``{prefix}_hat`` columns.
    prefix : str
        Column prefix (e.g., ``"target"`` or ``"color"``).
    field : {"true", "hat", "error", "abs_error"}
        Which derived field to return.

    Returns
    -------
    numpy.ndarray
        Array of values for the requested field.
    """
    true_col = f"{prefix}_true"
    hat_col = f"{prefix}_hat"
    if field == "true":
        return df[true_col].to_numpy(dtype=float)
    if field == "hat":
        return df[hat_col].to_numpy(dtype=float)
    if field == "error":
        return df[hat_col].to_numpy(dtype=float) - df[true_col].to_numpy(dtype=float)
    if field == "abs_error":
        return np.abs(df[hat_col].to_numpy(dtype=float) - df[true_col].to_numpy(dtype=float))
    raise ValueError(f"Unknown field: {field!r}")


def _default_color_param(params: Sequence[str]) -> str:
    """Choose a default color parameter (prefers beta-like names).

    Parameters
    ----------
    params : Sequence[str]
        Parameter names.

    Returns
    -------
    str
        Selected parameter name.
    """
    for p in params:
        if "beta" in p.lower():
            return p
    return params[0]


def plot_recovery_colored_by_other_param(
    *,
    df: pd.DataFrame,
    target_param: str,
    color_param: str,
    out_dir: str | Path,
    x_field: ValueField = "true",
    y_field: ValueField = "hat",
    color_field: ValueField = "true",
    color_min: float | None = None,
    color_max: float | None = None,
    color_quantile: float | None = None,
    color_quantile_side: Literal["upper", "lower"] = "upper",
    alpha: float = 0.6,
    max_points: int = 50_000,
    s: int = 10,
    cmap: str = "viridis",
    identity_line: bool = True,
    title: str | None = None,
    filename: str | None = None,
) -> Path:
    """Plot recovery for one parameter, colored by another parameter's values.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-form recovery records with columns ``rep``, ``subject_id``, ``param``,
        ``true``, and ``hat``.
    target_param : str
        Parameter whose recovery is plotted on the axes.
    color_param : str
        Parameter whose values control the color of points.
    out_dir : str or pathlib.Path
        Output directory for the plot.
    x_field : {"true", "hat", "error", "abs_error"}, optional
        Value shown on the x-axis for the target parameter.
    y_field : {"true", "hat", "error", "abs_error"}, optional
        Value shown on the y-axis for the target parameter.
    color_field : {"true", "hat", "error", "abs_error"}, optional
        Value used for color encoding of the other parameter.
    color_min, color_max : float or None, optional
        Optional range filter for the color values (keep only rows within).
    color_quantile : float or None, optional
        Optional quantile filter for color values (e.g., 0.9 keeps the top 10%).
    color_quantile_side : {"upper", "lower"}, optional
        Whether the quantile filter keeps the upper or lower tail.
    alpha : float, optional
        Scatter alpha.
    max_points : int, optional
        Maximum number of points to plot (uniform random sample if exceeded).
    s : int, optional
        Marker size.
    cmap : str, optional
        Matplotlib colormap name.
    identity_line : bool, optional
        Whether to draw y=x (useful for true vs hat).
    title : str or None, optional
        Optional title override.
    filename : str or None, optional
        Optional filename override.

    Returns
    -------
    pathlib.Path
        Path to the saved plot image.

    Examples
    --------
    Color recovery of ``alpha`` by ``beta``:

    >>> # plot_recovery_colored_by_other_param(  # doctest: +SKIP
    ... #     df=records,
    ... #     target_param="alpha",
    ... #     color_param="beta",
    ... #     out_dir="plots",
    ... # )
    """
    _require_columns(df, ["param", "true", "hat"])
    merged = _merge_target_and_color(df=df, target_param=target_param, color_param=color_param)

    x = _extract_value(merged, prefix="target", field=x_field)
    y = _extract_value(merged, prefix="target", field=y_field)
    c = _extract_value(merged, prefix="color", field=color_field)

    if color_min is not None:
        mask = c >= float(color_min)
        x, y, c = x[mask], y[mask], c[mask]
    if color_max is not None:
        mask = c <= float(color_max)
        x, y, c = x[mask], y[mask], c[mask]
    if color_quantile is not None:
        q = float(color_quantile)
        threshold = float(np.quantile(c, q))
        if color_quantile_side == "upper":
            mask = c >= threshold
        else:
            mask = c <= threshold
        x, y, c = x[mask], y[mask], c[mask]

    if len(x) == 0:
        raise ValueError("No points to plot after filtering.")

    if len(x) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(x), size=max_points, replace=False)
        x, y, c = x[idx], y[idx], c[idx]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        filename
        if filename is not None
        else f"recovery_{target_param}_colored_by_{color_param}.png"
    )

    plt.figure()
    sc = plt.scatter(x, y, c=c, alpha=alpha, s=s, cmap=cmap)
    if identity_line and x_field == "true" and y_field == "hat":
        lo = float(min(np.min(x), np.min(y)))
        hi = float(max(np.max(x), np.max(y)))
        pad = 0.02 * (hi - lo + 1e-12)
        lo -= pad
        hi += pad
        plt.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
        plt.xlim(lo, hi)
        plt.ylim(lo, hi)

    plt.xlabel(f"{target_param} ({x_field})")
    plt.ylabel(f"{target_param} ({y_field})")
    plot_title = title or f"{target_param} recovery colored by {color_param} ({color_field})"
    plt.title(plot_title)
    cb = plt.colorbar(sc)
    cb.set_label(f"{color_param} ({color_field})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_recovery_colored_by_param(
    *,
    df: pd.DataFrame,
    out_dir: str | Path,
    color_param: str | None = None,
    target_params: Sequence[str] | None = None,
    x_field: ValueField = "true",
    y_field: ValueField = "hat",
    color_field: ValueField = "true",
    alpha: float = 0.6,
    max_points: int = 50_000,
    color_min: float | None = None,
    color_max: float | None = None,
    color_quantile: float | None = None,
    color_quantile_side: Literal["upper", "lower"] = "upper",
) -> list[Path]:
    """Plot recovery for multiple parameters colored by one other parameter.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-form recovery records.
    out_dir : str or pathlib.Path
        Output directory for plots.
    color_param : str or None, optional
        Parameter used for coloring. If None, uses a heuristic (prefers a name
        containing ``"beta"``).
    target_params : Sequence[str] or None, optional
        Parameters to plot. If None, plots all parameters except ``color_param``.
    x_field, y_field, color_field : {"true", "hat", "error", "abs_error"}, optional
        Value fields for the target axes and color encoding.
    alpha, max_points : float, int, optional
        Scatter alpha and maximum points per plot.
    color_min, color_max : float or None, optional
        Optional range filter for the color values.
    color_quantile : float or None, optional
        Optional quantile filter for color values.
    color_quantile_side : {"upper", "lower"}, optional
        Whether the quantile filter keeps the upper or lower tail.

    Returns
    -------
    list[pathlib.Path]
        Paths to the saved plot images.
    """
    _require_columns(df, ["param", "true", "hat"])
    params = sorted({str(p) for p in df["param"].unique()})
    if not params:
        return []

    color_param = str(color_param) if color_param is not None else _default_color_param(params)
    if color_param not in params:
        raise ValueError(f"color_param {color_param!r} not found in df.param.")

    if target_params is None:
        target_params = [p for p in params if p != color_param]
    if not target_params:
        return []

    out_paths: list[Path] = []
    for target in target_params:
        out_paths.append(
            plot_recovery_colored_by_other_param(
                df=df,
                target_param=str(target),
                color_param=color_param,
                out_dir=out_dir,
                x_field=x_field,
                y_field=y_field,
                color_field=color_field,
                alpha=alpha,
                max_points=max_points,
                color_min=color_min,
                color_max=color_max,
                color_quantile=color_quantile,
                color_quantile_side=color_quantile_side,
            )
        )
    return out_paths
