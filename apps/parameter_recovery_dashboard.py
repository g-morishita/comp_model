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


def _plan_fields_from_manifest(man: dict[str, Any]) -> dict[str, Any]:
    ps = (man.get("plan_summary") or {}) if isinstance(man, dict) else {}
    return {
        "n_subjects": ps.get("n_subjects"),
        "n_blocks_unique": ps.get("n_blocks_unique"),
        "trials_per_subject_mean": ps.get("total_trials_per_subject_mean"),
        "trials_per_subject_min": ps.get("total_trials_per_subject_min"),
        "trials_per_subject_max": ps.get("total_trials_per_subject_max"),
    }


def build_run_index(runs: list[RunEntry]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in runs:
        cfg = load_config(str(r.out_dir)) or {}
        man = load_manifest(str(r.out_dir)) or {}

        git_commit = _get_any(man, ["comp_model_impl_git_commit", "git_commit"])
        git_branch = _get_any(man, ["comp_model_impl_git_branch", "git_branch"])
        git_dirty = _get_any(man, ["comp_model_impl_git_dirty", "git_dirty"])

        plan_fields = _plan_fields_from_manifest(man)

        rows.append(
            {
                "run_id": r.run_id,
                "time": format_time(r.mtime),
                "git_commit": (git_commit[:12] if isinstance(git_commit, str) else git_commit),
                "git_branch": git_branch,
                "git_dirty": git_dirty,
                "model": man.get("model"),
                "estimator": man.get("estimator"),
                "generator": man.get("generator"),
                "n_reps": cfg.get("n_reps"),
                "seed": cfg.get("seed"),
                "plan_path": cfg.get("plan_path"),
                **plan_fields,
                "out_dir": str(r.out_dir),
            }
        )

    df = pd.DataFrame(rows)
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
        "n_subjects",
        "n_blocks_unique",
        "trials_per_subject_mean",
        "plan_path",
        "out_dir",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


def apply_search_filter(df: pd.DataFrame, query: str) -> pd.DataFrame:
    q = (query or "").strip().lower()
    if not q:
        return df

    hay_cols = [
        "run_id",
        "git_commit",
        "git_branch",
        "model",
        "estimator",
        "generator",
        "plan_path",
        "out_dir",
        "n_subjects",
        "n_blocks_unique",
    ]
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
    root = Path(
        st.sidebar.text_input(
            "Output root folder", value=os.environ.get("PR_OUTPUT_ROOT", "outputs")
        )
    ).expanduser()
    st.sidebar.caption("This folder should contain unique run subfolders.")

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

    # ---- Main: run table ----
    st.subheader("Runs")
    st.dataframe(filtered, use_container_width=True, height=320)

    if filtered.empty:
        st.warning("No runs match your search.")
        return

    run_ids = filtered["run_id"].tolist()
    selected_run_id = st.selectbox("Select a run", options=run_ids, index=0)
    selected = next(r for r in runs if r.run_id == selected_run_id)
    out_dir = selected.out_dir

    # ---- Load run artifacts ----
    cfg = load_config(str(out_dir))
    manifest = load_manifest(str(out_dir))
    metrics = load_metrics(str(out_dir))
    records = load_records(str(out_dir))
    diags = load_diags(str(out_dir))

    # ---- Header / downloads ----
    st.divider()
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])

    c1.subheader(f"Selected run: {selected.run_id}")
    c1.caption(f"{format_time(selected.mtime)} · {out_dir}")

    plan_file = _get_any(manifest, ["plan_file_copied"]) if manifest else None
    plan_local = (out_dir / plan_file) if plan_file else None
    if plan_local and plan_local.exists() and plan_local.is_file():
        c5.download_button(
            "Download plan",
            data=plan_local.read_bytes(),
            file_name=plan_local.name,
            mime="text/yaml" if plan_local.suffix.lower() in (".yml", ".yaml") else "application/json",
        )
    else:
        c5.caption("Plan: (not copied)")

    if (out_dir / RECORDS_PARQUET).exists():
        c2.download_button(
            "Records (parquet)",
            data=(out_dir / RECORDS_PARQUET).read_bytes(),
            file_name=RECORDS_PARQUET,
            mime="application/octet-stream",
        )
    elif (out_dir / RECORDS_CSV).exists():
        c2.download_button(
            "Records (csv)",
            data=(out_dir / RECORDS_CSV).read_bytes(),
            file_name=RECORDS_CSV,
            mime="text/csv",
        )

    if (out_dir / METRICS_NAME).exists():
        c3.download_button(
            "Metrics (csv)",
            data=(out_dir / METRICS_NAME).read_bytes(),
            file_name=METRICS_NAME,
            mime="text/csv",
        )

    if (out_dir / DIAGS_NAME).exists():
        c4.download_button(
            "Diagnostics (jsonl)",
            data=(out_dir / DIAGS_NAME).read_bytes(),
            file_name=DIAGS_NAME,
            mime="application/json",
        )

    # ---- Settings: key facts up front ----
    st.divider()
    st.header("Settings (important)")

    created_at = _get_any(manifest, ["created_at"]) if manifest else None
    model_name = _get_any(manifest, ["model"]) if manifest else None
    estimator_name = _get_any(manifest, ["estimator"]) if manifest else None
    generator_name = _get_any(manifest, ["generator"]) if manifest else None
    git_commit = _get_any(manifest, ["comp_model_impl_git_commit", "git_commit"]) if manifest else None
    git_branch = _get_any(manifest, ["comp_model_impl_git_branch", "git_branch"]) if manifest else None
    git_dirty = _get_any(manifest, ["comp_model_impl_git_dirty", "git_dirty"]) if manifest else None

    plan_path = (cfg or {}).get("plan_path") if cfg else None
    n_reps = (cfg or {}).get("n_reps") if cfg else None
    seed = (cfg or {}).get("seed") if cfg else None

    save_format = None
    try:
        save_format = (cfg or {}).get("output", {}).get("save_format")
    except Exception:
        save_format = None

    ps = (manifest or {}).get("plan_summary", {}) if manifest else {}
    n_subjects = ps.get("n_subjects")
    n_blocks_unique = ps.get("n_blocks_unique")
    tmin = ps.get("total_trials_per_subject_min")
    tmean = ps.get("total_trials_per_subject_mean")
    tmax = ps.get("total_trials_per_subject_max")
    canonical_blocks = ps.get("canonical_block_ids") or []
    canonical_trials = ps.get("canonical_trials_by_block") or []

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Subjects", str(n_subjects) if n_subjects is not None else "—")
    p2.metric("Unique blocks", str(n_blocks_unique) if n_blocks_unique is not None else "—")
    p3.metric("Trials / subject (mean)", f"{tmean:.2f}" if isinstance(tmean, (int, float)) else "—")
    p4.metric("Trials / subject (min–max)", f"{tmin}–{tmax}" if (tmin is not None and tmax is not None) else "—")

    if canonical_blocks and canonical_trials and len(canonical_blocks) == len(canonical_trials):
        with st.expander("Canonical block schedule (from first subject)"):
            st.dataframe(
                pd.DataFrame({"block_id": canonical_blocks, "n_trials": canonical_trials}),
                use_container_width=True,
            )

    blocks_table = ps.get("blocks_table") or []
    if blocks_table:
        with st.expander("Plan audit: per-block n_trials across subjects"):
            st.dataframe(pd.DataFrame(blocks_table), use_container_width=True)

    sA, sB, sC, sD = st.columns(4)
    sA.metric("Created", created_at or "—")
    sB.metric("Git commit", (git_commit[:12] if isinstance(git_commit, str) else (git_commit or "—")))
    sC.metric("Branch", git_branch or "—")
    sD.metric("Dirty", str(git_dirty) if git_dirty is not None else "—")

    st.markdown(
        f"""
**Model:** `{model_name or "—"}`  
**Estimator:** `{estimator_name or "—"}`  
**Generator:** `{generator_name or "—"}`  
**Plan path in config:** `{plan_path or "—"}`  
**n_reps / seed / format:** `{n_reps or "—"}` / `{seed or "—"}` / `{save_format or "—"}`
"""
    )

    if plan_local and plan_local.exists() and plan_local.is_file():
        with st.expander("Show copied plan file"):
            lang = "yaml" if plan_local.suffix.lower() in (".yml", ".yaml") else "json"
            st.code(plan_local.read_text(encoding="utf-8"), language=lang)

    with st.expander("Show full manifest JSON"):
        st.json(manifest or {})

    with st.expander("Show full config JSON"):
        st.json(cfg or {})

    # ---- Metrics ----
    st.divider()
    st.header("Metrics")
    if metrics.empty:
        st.warning("Metrics file is missing or empty.")
    else:
        st.dataframe(metrics, use_container_width=True)

    # ---- Records & Plot ----
    st.divider()
    st.header("Scatter Plot")

    if records.empty:
        st.warning("Records file is missing or empty.")
        return

    required_cols = {"rep", "subject_id", "param", "true", "hat"}
    if not required_cols.issubset(records.columns):
        st.error(f"Records missing required columns: {sorted(required_cols - set(records.columns))}")
        return

    # ---- Selection controls at the TOP of the plot ----
    st.subheader("Selection")
    ctrl1, ctrl2, ctrl3, ctrl4, ctrl5, ctrl6 = st.columns([1.2, 1.0, 1.0, 1.0, 1.1, 1.2])

    params = sorted(records["param"].dropna().unique().tolist())
    param_sel = ctrl1.selectbox("Param", options=["(all)"] + params)

    reps = sorted(records["rep"].dropna().unique().tolist())
    rep_sel = ctrl2.selectbox("Rep", options=["(all)"] + [str(r) for r in reps])

    color_by = ctrl3.selectbox("Color by", options=["error", "abs_error", "true", "hat"])
    alpha = ctrl4.slider("Opacity", min_value=0.05, max_value=1.0, value=0.35, step=0.05)

    plot_size = ctrl5.slider("Plot size (square)", min_value=350, max_value=900, value=600, step=50)
    max_points = int(
        ctrl6.number_input("Max points", min_value=1000, max_value=2_000_000, value=200_000, step=10_000)
    )

    df = records.copy().dropna(subset=["true", "hat"])

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

    brush = alt.selection_interval(name="select_zone")

    summary = (
        alt.Chart(df_plot)
        .transform_filter(brush)
        .transform_aggregate(n="count()")
        .mark_text(align="left", baseline="middle", fontSize=14)
        .encode(text=alt.Text("n:Q", format=".0f"))
        .properties(width=plot_size, height=30, title="Selected points (drag a box on the scatter plot)")
    )

    scatter = (
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
        .add_params(brush)
        .properties(width=plot_size, height=plot_size)
    )

    st.altair_chart(alt.vconcat(summary, scatter), use_container_width=False)

    # =============================
    # Parameter correlations (NEW)
    # =============================
    st.divider()
    st.header("Parameter correlations")

    df_corr = records.copy()
    df_corr = df_corr.dropna(subset=["rep", "subject_id", "param", "true", "hat"])
    if df_corr.empty:
        st.warning("Not enough data to compute correlations.")
    else:
        df_corr["error"] = df_corr["hat"] - df_corr["true"]
        df_corr["abs_error"] = df_corr["error"].abs()

        cA, cB, cC, cD, cE = st.columns([1.4, 1.0, 1.4, 1.1, 1.1])

        value_choice = cA.selectbox(
            "Values to correlate",
            options=["hat (estimated)", "true", "error (hat-true)", "abs_error"],
            index=0,
        )
        method_choice = cB.selectbox("Method", options=["spearman", "pearson"], index=0)

        rep_mode = cC.selectbox(
            "Rep aggregation",
            options=[
                "Use selected rep (if any; else pool all)",
                "Pool all reps (rep × subject rows)",
                "Average across reps per subject",
            ],
            index=0,
        )

        min_obs = int(
            cD.number_input("Min paired obs", min_value=2, max_value=10_000, value=5, step=1)
        )

        heatmap_size = int(cE.slider("Heatmap size", min_value=350, max_value=900, value=550, step=50))

        value_col = {
            "hat (estimated)": "hat",
            "true": "true",
            "error (hat-true)": "error",
            "abs_error": "abs_error",
        }[value_choice]

        # Wide table: rows = (rep, subject_id), cols = param
        wide = df_corr.pivot_table(
            index=["rep", "subject_id"],
            columns="param",
            values=value_col,
            aggfunc="mean",
        )

        # Apply rep handling
        if rep_mode == "Use selected rep (if any; else pool all)":
            if rep_sel != "(all)":
                try:
                    wide = wide.xs(int(rep_sel), level="rep", drop_level=True)
                except Exception:
                    pass  # fallback to pooled
        elif rep_mode == "Average across reps per subject":
            wide = wide.groupby(level="subject_id").mean()

        # Drop all-NaN + constants
        wide = wide.dropna(axis=1, how="all")
        const_cols = [c for c in wide.columns if wide[c].nunique(dropna=True) <= 1]
        if const_cols:
            st.caption(f"Dropped constant params: {', '.join(map(str, const_cols))}")
            wide = wide.drop(columns=const_cols, errors="ignore")

        if wide.shape[1] < 2:
            st.warning("Need at least 2 varying parameters to compute correlations.")
        else:
            corr = wide.corr(method=method_choice, min_periods=min_obs)

            # --- FIX: ensure distinct names so reset_index doesn't create duplicate columns
            corr = corr.copy()
            corr.index = corr.index.astype(str)
            corr.columns = corr.columns.astype(str)
            corr.index.name = "param_x"
            corr.columns.name = "param_y"

            corr_long = (
                corr.stack()
                .rename("corr")
                .reset_index()   # now produces columns: param_x, param_y, corr
                .dropna(subset=["corr"])
            )

            heat = (
                alt.Chart(corr_long)
                .mark_rect()
                .encode(
                    x=alt.X("param_x:N", sort=list(corr.columns), title=""),
                    y=alt.Y("param_y:N", sort=list(corr.columns), title=""),
                    color=alt.Color("corr:Q", scale=alt.Scale(domain=[-1, 1]), title=f"{method_choice} r"),
                    tooltip=[
                        alt.Tooltip("param_x:N"),
                        alt.Tooltip("param_y:N"),
                        alt.Tooltip("corr:Q", format=".3f"),
                    ],
                )
                .properties(width=heatmap_size, height=heatmap_size)
            )

            txt = (
                alt.Chart(corr_long)
                .mark_text(baseline="middle", fontSize=10)
                .encode(
                    x=alt.X("param_x:N", sort=list(corr.columns)),
                    y=alt.Y("param_y:N", sort=list(corr.columns)),
                    text=alt.Text("corr:Q", format=".2f"),
                )
            )

            st.altair_chart(heat + txt, use_container_width=False)

            st.download_button(
                "Download correlation matrix (csv)",
                data=corr.to_csv().encode("utf-8"),
                file_name=f"param_corr_{value_col}_{method_choice}.csv",
                mime="text/csv",
            )

            tri_mask = np.triu(np.ones_like(corr.to_numpy(), dtype=bool), k=1)
            pairs = (
                corr.where(tri_mask)
                .stack()
                .rename("corr")
                .reset_index()  # columns: param_x, param_y, corr
            )

            if not pairs.empty:
                pairs["abs_corr"] = pairs["corr"].abs()
                pairs = pairs.sort_values("abs_corr", ascending=False)
                st.subheader("Top correlated parameter pairs")
                st.dataframe(pairs.head(30), use_container_width=True)

            st.subheader("Pair scatter (optional)")
            p1c, p2c, p3c = st.columns([1.0, 1.0, 1.0])
            cols = list(corr.columns)

            x_param = p1c.selectbox("X", options=cols, index=0)
            y_param = p2c.selectbox("Y", options=cols, index=(1 if len(cols) > 1 else 0))
            point_alpha = p3c.slider("Opacity (pair)", min_value=0.05, max_value=1.0, value=0.35, step=0.05)

            df_pair = wide[[x_param, y_param]].dropna().reset_index()

            tooltips = []
            if "rep" in df_pair.columns:
                tooltips.append(alt.Tooltip("rep:Q"))
            if "subject_id" in df_pair.columns:
                tooltips.append(alt.Tooltip("subject_id:N"))
            tooltips += [
                alt.Tooltip(f"{x_param}:Q", format=".6g"),
                alt.Tooltip(f"{y_param}:Q", format=".6g"),
            ]

            pair_chart = (
                alt.Chart(df_pair)
                .mark_circle(opacity=point_alpha)
                .encode(
                    x=alt.X(f"{x_param}:Q", title=x_param),
                    y=alt.Y(f"{y_param}:Q", title=y_param),
                    tooltip=tooltips,
                )
                .properties(
                    width=min(800, heatmap_size + 150),
                    height=min(600, heatmap_size),
                )
            )
            st.altair_chart(pair_chart, use_container_width=False)

    # ---- Records preview ----
    st.subheader("Records (preview)")
    st.dataframe(df.head(5000), use_container_width=True)

    st.divider()
    st.header("Fit Diagnostics (optional)")
    if diags.empty:
        st.caption("No diagnostics file found (or empty).")
    else:
        st.dataframe(diags, use_container_width=True)


if __name__ == "__main__":
    main()
