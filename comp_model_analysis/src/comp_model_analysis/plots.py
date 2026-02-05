"""Plot helpers for analysis utilities (optional dependency: matplotlib)."""

from __future__ import annotations

import numpy as np


def plot_profile_1d(
    *,
    grid: np.ndarray,
    ll: np.ndarray,
    param: str,
    ax=None,
    true_value: float | None = None,
):
    """Plot a 1D profile likelihood curve."""
    import matplotlib.pyplot as plt  # optional dependency

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(grid, ll, marker="o", linewidth=1.5)
    if true_value is not None:
        ax.axvline(float(true_value), color="k", linestyle="--", linewidth=1)
    ax.set_xlabel(str(param))
    ax.set_ylabel("Best log-likelihood")
    return ax


def plot_profile_2d(
    *,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    ll: np.ndarray,
    param_x: str,
    param_y: str,
    ax=None,
):
    """Plot a 2D likelihood slice as a heatmap."""
    import matplotlib.pyplot as plt  # optional dependency

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(4, 3))
    im = ax.imshow(
        ll,
        origin="lower",
        aspect="auto",
        extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]],
    )
    ax.set_xlabel(str(param_x))
    ax.set_ylabel(str(param_y))
    ax.set_title("2D likelihood slice")
    plt.colorbar(im, ax=ax, label="Best log-likelihood")
    return ax
