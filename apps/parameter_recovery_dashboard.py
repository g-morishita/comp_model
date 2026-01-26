from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Files / conventions
# -----------------------------
DEFAULT_ROOT = os.environ.get("PR_OUTPUT_ROOT", "outputs")

CONFIG_NAME = "parameter_recovery_config.json"
METRICS_NAME = "parameter_recovery_metrics.csv"
RECORDS_CSV = "parameter_recovery_records.csv"
RECORDS_PARQUET = "parameter_recovery_records.parquet"
DIAGS_NAME = "parameter_recovery_fit_diagnostics.jsonl"
MANIFEST_NAME = "run_manifest.json"


# -----------------------------
# Types
# -----------------------------
@dataclass(frozen=True)
class RunEntry:
    run_id: str
    out_dir: Path
    mtime: float


# -----------------------------
# Helpers: filesystem + parsing
# -----------------------------
def format_time(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _safe_read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_jsonl(path: Path, max_rows: int = 2000) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _find_runs(root: Path) -> list[RunEntry]:
    """
    A "run" is a directory that contains:
      - parameter_recovery_config.json
      - parameter_recovery_metrics.csv
    """
    if not root.exists():
        return []

    runs: list[RunEntry] = []
    for cfg_path in root.rglob(CONFIG_NAME):
        out_dir = cfg_path.parent
        metrics_path = out_dir / METRICS_NAME
        if not metrics_path.exists():
            continue

        mtime = max(cfg_path.stat().st_mtime, metrics_path.stat().st_mtime)
        runs.append(RunEntry(run_id=out_dir.name, out_dir=out_dir, mtime=mtime))

    runs.sort(key=lambda r: r.mtime, reverse=True)
    return runs


# -----------------------------
# Cached loaders
# -----------------------------
@st.cache_data(show_spinner=False)
def load_config(out_dir: str) -> dict[str, Any] | None:
    p = Path(out_dir) / CONFIG_NAME
    if not p.exists():
        return None
    return _safe_read_json(p)


@st.cache_data(show_spinner=False)
def load_manifest(out_dir: str) -> dict[str, Any] | None:
    p = Path(out_dir) / MANIFEST_NAME
    if not p.exists():
        return None
    return _safe_read_json(p)


@st.cache_data(show_spinner=False)
def load_metrics(out_dir: str) -> pd.DataFrame:
    p = Path(out_dir) / METRICS_NAME
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_records(out_dir: str) -> pd.DataFrame:
    od = Path(out_dir)
    pq = od / RECORDS_PARQUET
    csv = od / RECORDS_CSV
    try:
        if pq.exists():
            return pd.read_parquet(pq)
        if csv.exists():
            return pd.read_csv(csv)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_diags(out_dir: str) -> pd.DataFrame:
    p = Path(out_dir) / DIAGS_NAME
    if not p.exists():
        return pd.DataFrame()
    return _read_jsonl(p, max_rows=5000)


# -----------------------------
# Run indexing / search
# -----------------------------
def _get_any(d: dict[str, Any] | None, keys: list[str]) -> Any:
    if not d:
        return None
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def build_run_index(runs: list[RunEntry]) -> pd.DataFrame:
    """
    Create a searchable "index" table of runs with relevant fields.
    """
    rows: list[dict[str, Any]] = []
    for r in runs:
        cfg = load_config(str(r.out_dir)) or {}
        man = load_manifest(str(r.out_dir)) or {}

        # tolerate either "comp_model_impl_git_commit" (module-specific) or generic "git_commit"
        git_commit = _get_any(man, ["comp_model_impl_git_commit", "git_commit"])
        git_branch = _get_any(man, ["comp_model_impl_git_branch", "git_branch"])
        git_dirty = _get_any(man, ["comp_model_impl_git_dirty", "git_dirty"])

        rows.append(
            {
                "run_id": r.run_id,
                "time": format_time(r.mtime),
                "out_dir": str(r.out_dir),
                "git_commit": git_commit,
                "git_branch": git_branch,
                "git_dirty": git_dirty,
                "model": man.get("model"),
                "estimator": man.get("estimator"),
                "generator": man.get("generator"),
                "plan_path": cfg.get("plan_path"),
                "n_reps": cfg.get("n_reps"),
                "seed": cfg.get("seed"),
            }
        )

    df = pd.DataFrame(rows)
    # keep stable column order
    cols = [
        "run_id",
        "time",
        "git_commit",
        "git_branch",
        "git_dirty",
        "model",
        "estimator",
        "generator",
        "n_reps",
        "seed",
        "plan_path",
        "out_dir",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


def apply_search_filter(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Very simple search: case-insensitive substring across key columns.
    """
    q = (query or "").strip().lower()
    if not q:
        return df

    hay_cols = ["run_id", "git_commit", "git_branch", "model", "estimator", "generator", "plan_path", "out_dir"]
    mask = pd.Series(False, index=df.index)
    for c in hay_cols:
        s = df[c].astype(str).str.lower()
        mask |= s.str.contains(q, na=False)

    return df[mask]


# -----------------------------
# UI helpers
# -----------------------------
def summarize_df(df: pd.DataFrame) -> dict[str, float]:
    df = df.dropna(subset=["true", "hat"])
    if df.empty:
        return {"n": 0.0, "rmse": np.nan, "mae": np.nan, "corr": np.nan}
    err = df["hat"] - df["true"]
    rmse = float(np.sqrt(np.mean(np.square(err))))
    mae = float(np.mean(np.abs(err)))
    corr = float(np.corrcoef(df["true"], df["hat"])[0, 1]) if len(df) > 1 else np.nan
    return {"n": float(len(df)), "rmse": rmse, "mae": mae, "corr": corr}


def main() -> None:
    st.set_page_config(page_title="Parameter Recovery Dashboard", layout="wide")
    st.title("Parameter Recovery Dashboard")

    # ---- Sidebar: root + search ----
    root = Path(st.sidebar.text_input("Output root folder", value=DEFAULT_ROOT)).expanduser()
    st.sidebar.caption("This folder should contain run subfolders (unique per run).")

    runs = _find_runs(root)
    if not runs:
        st.info(
            f"No runs found under '{root}'. "
            f"Expected each run to contain '{CONFIG_NAME}' and '{METRICS_NAME}'."
        )
        return

    run_index = build_run_index(runs)

    st.sidebar.subheader("Search runs")
    query = st.sidebar.text_input("Search (substring)", value="")
    filtered = apply_search_filter(run_index, query)

    st.sidebar.caption(f"Runs: {len(filtered)} / {len(run_index)}")

    # ---- Main: table of runs ----
    st.subheader("Runs")
    st.dataframe(filtered, use_container_width=True, height=320)

    # Select a run from filtered results
    if filtered.empty:
        st.warning("No runs match your search.")
        return

    run_ids = filtered["run_id"].tolist()
    default_run = run_ids[0]
    selected_run_id = st.selectbox("Select a run", options=run_ids, index=0)
    selected = next(r for r in runs if r.run_id == selected_run_id)
    out_dir = selected.out_dir

    # ---- Load selected run artifacts ----
    cfg = load_config(str(out_dir))
    manifest = load_manifest(str(out_dir))
    metrics = load_metrics(str(out_dir))
    records = load_records(str(out_dir))
    diags = load_diags(str(out_dir))

    # ---- Header / downloads ----
    st.divider()
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    c1.subheader(f"Selected run: {selected.run_id}")
    c1.caption(f"{format_time(selected.mtime)} · {out_dir}")

    if (out_dir / RECORDS_PARQUET).exists():
        c2.download_button(
            "Download records (parquet)",
            data=(out_dir / RECORDS_PARQUET).read_bytes(),
            file_name=RECORDS_PARQUET,
            mime="application/octet-stream",
        )
    elif (out_dir / RECORDS_CSV).exists():
        c2.download_button(
            "Download records (csv)",
            data=(out_dir / RECORDS_CSV).read_bytes(),
            file_name=RECORDS_CSV,
            mime="text/csv",
        )

    if (out_dir / METRICS_NAME).exists():
        c3.download_button(
            "Download metrics (csv)",
            data=(out_dir / METRICS_NAME).read_bytes(),
            file_name=METRICS_NAME,
            mime="text/csv",
        )

    if (out_dir / DIAGS_NAME).exists():
        c4.download_button(
            "Download diagnostics (jsonl)",
            data=(out_dir / DIAGS_NAME).read_bytes(),
            file_name=DIAGS_NAME,
            mime="application/json",
        )

    # ---- Settings ----
    st.divider()
    st.header("Settings")
    s1, s2 = st.columns(2)
    with s1:
        st.subheader("Config")
        if cfg is None:
            st.warning(f"Missing {CONFIG_NAME}")
        else:
            st.json(cfg)
    with s2:
        st.subheader("Manifest")
        if manifest is None:
            st.info(f"Missing {MANIFEST_NAME} (optional)")
        else:
            st.json(manifest)

    # ---- Metrics ----
    st.divider()
    st.header("Metrics")
    if metrics.empty:
        st.warning("Metrics file is missing or empty.")
    else:
        st.dataframe(metrics, use_container_width=True)

    # ---- Records & plots ----
    st.divider()
    st.header("Records & Plot")
    if records.empty:
        st.warning("Records file is missing or empty.")
        return

    required_cols = {"rep", "subject_id", "param", "true", "hat"}
    if not required_cols.issubset(records.columns):
        st.error(f"Records missing required columns: {sorted(required_cols - set(records.columns))}")
        return

    # Filters for selected run only
    sidebar = st.sidebar
    sidebar.subheader("Filters (selected run)")

    params = sorted(records["param"].dropna().unique().tolist())
    param_sel = sidebar.selectbox("Param", options=["(all)"] + params)

    reps = sorted(records["rep"].dropna().unique().tolist())
    rep_sel = sidebar.selectbox("Rep", options=["(all)"] + [str(r) for r in reps])

    color_by = sidebar.selectbox("Color by", options=["error", "abs_error", "true", "hat"])
    alpha = sidebar.slider("Point opacity", min_value=0.05, max_value=1.0, value=0.35, step=0.05)
    max_points = int(
        sidebar.number_input("Max points (subsample)", min_value=1000, max_value=2_000_000, value=200_000, step=10_000)
    )

    df = records.copy()
    df = df.dropna(subset=["true", "hat"])

    if param_sel != "(all)":
        df = df[df["param"] == param_sel]
    if rep_sel != "(all)":
        df = df[df["rep"] == int(rep_sel)]

    df["error"] = df["hat"] - df["true"]
    df["abs_error"] = df["error"].abs()

    if len(df) > max_points:
        df_plot = df.sample(n=max_points, random_state=0)
    else:
        df_plot = df

    stats = summarize_df(df)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("N", int(stats["n"]))
    k2.metric("RMSE", f"{stats['rmse']:.4g}" if np.isfinite(stats["rmse"]) else "NA")
    k3.metric("MAE", f"{stats['mae']:.4g}" if np.isfinite(stats["mae"]) else "NA")
    k4.metric("Corr", f"{stats['corr']:.4g}" if np.isfinite(stats["corr"]) else "NA")

    st.subheader("True vs Estimated")
    chart = (
        alt.Chart(df_plot)
        .mark_circle(opacity=alpha)
        .encode(
            x=alt.X("true:Q", title="True"),
            y=alt.Y("hat:Q", title="Estimated"),
            color=alt.Color(f"{color_by}:Q", title=color_by),
            tooltip=[
                alt.Tooltip("rep:Q"),
                alt.Tooltip("subject_id:N"),
                alt.Tooltip("param:N"),
                alt.Tooltip("true:Q", format=".6g"),
                alt.Tooltip("hat:Q", format=".6g"),
                alt.Tooltip("error:Q", format=".6g"),
            ],
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Records (preview)")
    st.dataframe(df.head(5000), use_container_width=True)

    # ---- Diagnostics (optional) ----
    st.divider()
    st.header("Fit Diagnostics (optional)")
    if diags.empty:
        st.caption("No diagnostics file found (or empty).")
    else:
        st.dataframe(diags, use_container_width=True)


if __name__ == "__main__":
    main()
