"""Plotting utilities for parameter recovery.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Plotly is optional; only import when used
try:
    import plotly.express as px
except Exception:  # noqa: BLE001
    px = None


ColorMode = Literal["true", "hat", "error", "abs_error"]


def plot_parameter_recovery_scatter(
    *,
    df: pd.DataFrame,
    out_dir: str | Path,
    alpha: float = 0.6,
    max_points: int = 50_000,
) -> None:
    """
    For each parameter, save a scatter plot: true vs hat with identity line.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for param, g in df.groupby("param", sort=True):
        g = g.dropna(subset=["true", "hat"])
        if len(g) == 0:
            continue

        if len(g) > max_points:
            g = g.sample(n=max_points, random_state=0)

        x = g["true"].to_numpy(dtype=float)
        y = g["hat"].to_numpy(dtype=float)

        lo = float(min(np.min(x), np.min(y)))
        hi = float(max(np.max(x), np.max(y)))
        pad = 0.02 * (hi - lo + 1e-12)
        lo -= pad
        hi += pad

        plt.figure()
        plt.scatter(x, y, alpha=alpha, s=10)
        plt.plot([lo, hi], [lo, hi])  # identity
        plt.xlim(lo, hi)
        plt.ylim(lo, hi)
        plt.xlabel("True")
        plt.ylabel("Estimated")
        plt.title(f"Parameter recovery: {param}")
        plt.tight_layout()
        plt.savefig(out_dir / f"recovery_{param}.png", dpi=150)
        plt.close()


def plot_parameter_recovery_scatter_color(
    *,
    df: pd.DataFrame,
    out_dir: str | Path,
    color_by: ColorMode = "true",
    alpha: float = 0.6,
    max_points: int = 50_000,
) -> None:
    """
    Scatter true vs hat per parameter, with color encoding.

    color_by:
      - "true": color = true value
      - "hat": color = estimated value
      - "error": color = (hat - true)
      - "abs_error": color = |hat - true|
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for param, g in df.groupby("param", sort=True):
        g = g.dropna(subset=["true", "hat"])
        if len(g) == 0:
            continue

        if len(g) > max_points:
            g = g.sample(n=max_points, random_state=0)

        x = g["true"].to_numpy(dtype=float)
        y = g["hat"].to_numpy(dtype=float)

        if color_by == "true":
            c = x
            c_label = "True"
        elif color_by == "hat":
            c = y
            c_label = "Estimated"
        elif color_by == "error":
            c = y - x
            c_label = "Error (hat - true)"
        elif color_by == "abs_error":
            c = np.abs(y - x)
            c_label = "Abs error |hat - true|"
        else:
            raise ValueError(f"Unknown color_by: {color_by}")

        lo = float(min(np.min(x), np.min(y)))
        hi = float(max(np.max(x), np.max(y)))
        pad = 0.02 * (hi - lo + 1e-12)
        lo -= pad
        hi += pad

        plt.figure()
        sc = plt.scatter(x, y, c=c, alpha=alpha, s=10)
        plt.plot([lo, hi], [lo, hi])
        plt.xlim(lo, hi)
        plt.ylim(lo, hi)
        plt.xlabel("True")
        plt.ylabel("Estimated")
        plt.title(f"Parameter recovery: {param} (color by {color_by})")
        cb = plt.colorbar(sc)
        cb.set_label(c_label)
        plt.tight_layout()
        plt.savefig(out_dir / f"recovery_{param}_color_{color_by}.png", dpi=150)
        plt.close()


def plot_parameter_recovery_interactive(
    *,
    df: pd.DataFrame,
    out_path: str | Path,
    max_points: int = 200_000,
) -> None:
    """
    Interactive parameter recovery scatter using Plotly.
    Writes a self-contained HTML file.

    Requirements:
      pip install plotly
    """
    if px is None:
        raise ImportError("plotly is required for interactive plots. Install: pip install plotly")

    g = df.dropna(subset=["true", "hat"]).copy()
    if len(g) == 0:
        return

    if len(g) > max_points:
        g = g.sample(n=max_points, random_state=0)

    g["error"] = g["hat"] - g["true"]
    g["abs_error"] = np.abs(g["error"])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # We’ll create a facetable interactive plot:
    # - user can filter by param via Plotly legend/controls; best is to create a param dropdown
    # Plotly Express doesn't provide a dropdown directly, but facet_row/col can work.
    # Instead, simplest: make one plot with param as a dropdown via animation_frame.
    fig = px.scatter(
        g,
        x="true",
        y="hat",
        color="true",  # initial
        animation_frame="param",  # dropdown-like slider
        hover_data=["subject_id", "rep", "true", "hat", "error", "abs_error"],
        title="Parameter recovery (interactive): true vs estimated",
    )

    # Add identity line not per frame (Plotly limitation); but we can set equal scaling:
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    fig.write_html(str(out_path), include_plotlyjs="cdn")
