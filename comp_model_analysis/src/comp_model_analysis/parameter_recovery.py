"""Parameter recovery plotting utilities (optional dependency: matplotlib)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _as_path(p: str | Path | None) -> Path | None:
    if p is None:
        return None
    return p if isinstance(p, Path) else Path(p)


def _load_tables_from_run_dir(run_dir: Path) -> dict[str, Any]:
    import pandas as pd  # optional dependency

    records_path = run_dir / "parameter_recovery_records.csv"
    metrics_path = run_dir / "parameter_recovery_metrics.csv"
    pop_records_path = run_dir / "population_recovery_records.csv"
    pop_metrics_path = run_dir / "population_recovery_metrics.csv"

    out: dict[str, Any] = {
        "records": pd.read_csv(records_path) if records_path.exists() else None,
        "metrics": pd.read_csv(metrics_path) if metrics_path.exists() else None,
        "population_records": pd.read_csv(pop_records_path) if pop_records_path.exists() else None,
        "population_metrics": pd.read_csv(pop_metrics_path) if pop_metrics_path.exists() else None,
    }
    return out


def _ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _savefig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def _scatter_true_hat(df, *, title: str, out_path: Path, max_points: int, alpha: float) -> None:
    import matplotlib.pyplot as plt  # optional dependency
    import numpy as np

    params = sorted(df["param"].unique())
    n = len(params)
    if n == 0:
        return

    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

    for i, p in enumerate(params):
        ax = axes[i // ncols][i % ncols]
        sub = df[df["param"] == p]
        if len(sub) > max_points:
            sub = sub.sample(n=max_points, random_state=0)

        x = sub["true"].to_numpy()
        y = sub["hat"].to_numpy()
        ax.scatter(x, y, s=8, alpha=alpha)
        mn = float(min(x.min(), y.min()))
        mx = float(max(x.max(), y.max()))
        ax.plot([mn, mx], [mn, mx], color="k", linewidth=1)
        ax.set_title(str(p))
        ax.set_xlabel("true")
        ax.set_ylabel("hat")

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(title)
    _savefig(fig, out_path)
    plt.close(fig)


def _bar_metrics(metrics, *, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt  # optional dependency
    import numpy as np

    if metrics is None or metrics.empty:
        return

    agg = metrics.groupby("param", as_index=False).agg(
        corr=("corr", "mean"),
        rmse=("rmse", "mean"),
        bias=("bias", "mean"),
        mae=("mae", "mean"),
        median_abs_error=("median_abs_error", "mean"),
    )
    params = agg["param"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    x = np.arange(len(params))

    axes[0].bar(x, agg["corr"].to_numpy())
    axes[0].set_title("corr (mean)")
    axes[0].set_xticks(x, params, rotation=45, ha="right")

    axes[1].bar(x, agg["rmse"].to_numpy())
    axes[1].set_title("rmse (mean)")
    axes[1].set_xticks(x, params, rotation=45, ha="right")

    axes[2].bar(x, agg["mae"].to_numpy())
    axes[2].set_title("mae (mean)")
    axes[2].set_xticks(x, params, rotation=45, ha="right")

    fig.suptitle(title)
    _savefig(fig, out_path)
    plt.close(fig)


def _corr_hist(metrics, *, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt  # optional dependency
    import numpy as np

    if metrics is None or metrics.empty:
        return

    # Histogram only meaningful if rep count > 2 per parameter
    counts = metrics.groupby("param")["rep"].nunique()
    if int(counts.max()) <= 2:
        return

    params = sorted(metrics["param"].unique())
    n = len(params)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for i, p in enumerate(params):
        ax = axes[i // ncols][i % ncols]
        sub = metrics[metrics["param"] == p]
        ax.hist(sub["corr"].to_numpy(), bins=10, range=(-1, 1))
        ax.set_title(str(p))
        ax.set_xlabel("corr")
        ax.set_ylabel("count")

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(title)
    _savefig(fig, out_path)
    plt.close(fig)


def plot_parameter_recovery(
    *,
    outputs: Any | None = None,
    run_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    max_points: int = 50000,
    scatter_alpha: float = 0.6,
) -> dict[str, Path]:
    """
    Plot parameter recovery for subject- and population-level records.

    Parameters
    ----------
    outputs:
        ParameterRecoveryOutputs-like object (duck-typed; must have ``records`` and ``metrics``).
    run_dir:
        Directory containing recovery CSVs (parameter_recovery_records.csv, etc.).
    out_dir:
        Output directory for saved images. Defaults to ``<run_dir>/plots`` or ``./plots``.
    max_points:
        Max points per scatter plot.
    scatter_alpha:
        Scatter alpha for true-vs-hat plots.

    Returns
    -------
    dict[str, Path]
        Mapping of plot keys to saved file paths.
    """
    import pandas as pd  # optional dependency

    if outputs is None and run_dir is None:
        raise ValueError("Provide either outputs or run_dir.")

    tables: dict[str, Any]
    if outputs is not None:
        tables = {
            "records": getattr(outputs, "records", None),
            "metrics": getattr(outputs, "metrics", None),
            "population_records": getattr(outputs, "population_records", None),
            "population_metrics": getattr(outputs, "population_metrics", None),
        }
        base_out_dir = _as_path(out_dir) or Path("plots")
    else:
        rd = _as_path(run_dir)
        if rd is None:
            raise ValueError("run_dir must be provided when outputs is None.")
        tables = _load_tables_from_run_dir(rd)
        base_out_dir = _as_path(out_dir) or (rd / "plots")

    _ensure_out_dir(base_out_dir)

    paths: dict[str, Path] = {}

    records = tables.get("records")
    metrics = tables.get("metrics")
    pop_records = tables.get("population_records")
    pop_metrics = tables.get("population_metrics")

    if isinstance(records, pd.DataFrame) and not records.empty:
        p = base_out_dir / "recovery_scatter.png"
        _scatter_true_hat(
            records,
            title="Parameter recovery (subject-level)",
            out_path=p,
            max_points=max_points,
            alpha=scatter_alpha,
        )
        paths["recovery_scatter"] = p

    if isinstance(metrics, pd.DataFrame) and not metrics.empty:
        p = base_out_dir / "recovery_metrics.png"
        _bar_metrics(metrics, title="Recovery metrics (mean over reps)", out_path=p)
        paths["recovery_metrics"] = p

        p = base_out_dir / "recovery_corr_hist.png"
        _corr_hist(metrics, title="Correlation across reps", out_path=p)
        if p.exists():
            paths["recovery_corr_hist"] = p

    if isinstance(pop_records, pd.DataFrame) and not pop_records.empty:
        p = base_out_dir / "population_recovery_scatter.png"
        _scatter_true_hat(
            pop_records,
            title="Population recovery",
            out_path=p,
            max_points=max_points,
            alpha=scatter_alpha,
        )
        paths["population_recovery_scatter"] = p

    if isinstance(pop_metrics, pd.DataFrame) and not pop_metrics.empty:
        p = base_out_dir / "population_recovery_metrics.png"
        _bar_metrics(pop_metrics, title="Population recovery metrics", out_path=p)
        paths["population_recovery_metrics"] = p

    return paths
