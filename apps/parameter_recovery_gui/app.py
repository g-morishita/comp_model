"""Streamlit GUI for parameter recovery runs."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import math
import numpy as np
import pandas as pd
import streamlit as st
try:
    import plotly.express as px
except Exception:  # pragma: no cover - optional dependency for results view
    px = None

try:
    import yaml
except ImportError:  # pragma: no cover - streamlit-only failure path
    st.error("PyYAML is required. Install with: pip install pyyaml")
    st.stop()

from comp_model_core.interfaces.model import SocialComputationalModel
from comp_model_core.plans.io import infer_load_conditions
from comp_model_impl.estimators import (
    BoxMLESubjectwiseEstimator,
    StanHierarchicalNUTSEstimator,
    TransformedMLESubjectwiseEstimator,
    WithinSubjectSharedDeltaTransformedMLEEstimator,
)
from comp_model_impl.estimators.stan.adapters.registry import resolve_stan_adapter
from comp_model_impl.generators.event_log import (
    EventLogAsocialGenerator,
    EventLogSocialPreChoiceGenerator,
)
from comp_model_impl.recovery.parameter.config import load_parameter_recovery_config
from comp_model_impl.recovery.parameter.analysis import compute_parameter_recovery_metrics
from comp_model_impl.recovery.parameter.run import run_parameter_recovery
from comp_model_impl.register import make_registry

DEFAULT_CONDITIONS = ["A", "B", "C"]

DIST_FAMILIES = ["norm", "lognorm", "beta", "gamma", "expon", "uniform", "constant"]

def _plan_mentions_demonstrator(plan_raw: dict[str, Any]) -> bool:
    subjects = plan_raw.get("subjects")
    if isinstance(subjects, dict):
        for blocks in subjects.values():
            if isinstance(blocks, list):
                for b in blocks:
                    if isinstance(b, dict) and b.get("demonstrator_type") is not None:
                        return True

    template = plan_raw.get("subject_template", {})
    if isinstance(template, dict):
        blocks = template.get("blocks", [])
        if isinstance(blocks, list):
            for b in blocks:
                if isinstance(b, dict) and b.get("demonstrator_type") is not None:
                    return True

    return False


def _default_action_sequence(n_trials: int, n_actions: int) -> str:
    return ",".join(str(i % n_actions) for i in range(n_trials))


def _parse_action_sequence(text: str) -> list[int]:
    out: list[int] = []
    if not text:
        return out
    for raw in text.replace(";", ",").split(","):
        token = raw.strip()
        if token == "":
            continue
        out.append(int(token))
    return out


def _default_scale(val: float) -> float:
    base = abs(float(val))
    if base < 1e-6:
        base = 1.0
    return 0.1 * base


def _base_param_from_prior_name(name: str) -> str:
    base = str(name)
    if base.startswith("mu_"):
        base = base[len("mu_") :]
    elif base.startswith("sd_"):
        base = base[len("sd_") :]
    for suffix in ("__shared", "__delta"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base


def _default_hyper_priors(model_obj: Any, required: list[str]) -> dict[str, dict[str, float | str]]:
    base_schema = getattr(model_obj, "param_schema", None)
    z_defaults: dict[str, float] = {}
    if base_schema is not None:
        try:
            z0 = base_schema.default_z()
            for i, p in enumerate(getattr(base_schema, "params", [])):
                z_defaults[str(getattr(p, "name", f"p{i}"))] = float(z0[i])
        except Exception:
            z_defaults = {}

    out: dict[str, dict[str, float | str]] = {}
    for name in required:
        lname = str(name).lower()
        if lname.startswith("mu_"):
            base = _base_param_from_prior_name(name)
            mu = 0.0 if name.endswith("__delta") else float(z_defaults.get(base, 0.0))
            out[name] = {"family": "normal", "mu": float(mu), "sigma": 1.0}
        elif lname.startswith("sd_"):
            out[name] = {"family": "half-normal", "sigma": 1.0}
        else:
            out[name] = {"family": "normal", "mu": 0.0, "sigma": 1.0}
    return out


def _dist_args_inputs(
    *,
    dist_name: str,
    param_name: str,
    default_val: float,
    key_prefix: str,
    label_prefix: str | None = None,
    layout: str = "stacked",
) -> dict[str, float]:
    prefix = "" if not label_prefix else f"{label_prefix} "
    if dist_name == "norm":
        if layout == "columns":
            loc_col, scale_col = st.columns(2)
            with loc_col:
                loc = st.number_input(
                    f"{prefix}loc",
                    value=float(default_val),
                    step=0.01,
                    key=f"{key_prefix}_loc",
                )
            with scale_col:
                scale = st.number_input(
                    f"{prefix}scale",
                    min_value=1e-6,
                    value=float(_default_scale(default_val)),
                    step=0.01,
                    key=f"{key_prefix}_scale",
                )
        else:
            loc = st.number_input(
                f"{param_name} loc",
                value=float(default_val),
                step=0.01,
                key=f"{key_prefix}_loc",
            )
            scale = st.number_input(
                f"{param_name} scale",
                min_value=1e-6,
                value=float(_default_scale(default_val)),
                step=0.01,
                key=f"{key_prefix}_scale",
            )
        return {"loc": float(loc), "scale": float(scale)}

    if dist_name == "lognorm":
        if layout == "columns":
            s_col, loc_col, scale_col = st.columns(3)
            with s_col:
                s = st.number_input(
                    f"{prefix}s",
                    min_value=1e-6,
                    value=0.5,
                    step=0.01,
                    key=f"{key_prefix}_s",
                )
            with loc_col:
                loc = st.number_input(
                    f"{prefix}loc",
                    value=0.0,
                    step=0.01,
                    key=f"{key_prefix}_loc",
                )
            with scale_col:
                scale = st.number_input(
                    f"{prefix}scale",
                    min_value=1e-6,
                    value=max(1e-6, float(default_val)),
                    step=0.01,
                    key=f"{key_prefix}_scale",
                )
        else:
            s = st.number_input(
                f"{param_name} s",
                min_value=1e-6,
                value=0.5,
                step=0.01,
                key=f"{key_prefix}_s",
            )
            loc = st.number_input(
                f"{param_name} loc",
                value=0.0,
                step=0.01,
                key=f"{key_prefix}_loc",
            )
            scale = st.number_input(
                f"{param_name} scale",
                min_value=1e-6,
                value=max(1e-6, float(default_val)),
                step=0.01,
                key=f"{key_prefix}_scale",
            )
        return {"s": float(s), "loc": float(loc), "scale": float(scale)}

    if dist_name == "beta":
        if layout == "columns":
            a_col, b_col = st.columns(2)
            with a_col:
                a = st.number_input(
                    f"{prefix}a",
                    min_value=1e-6,
                    value=2.0,
                    step=0.1,
                    key=f"{key_prefix}_a",
                )
            with b_col:
                b = st.number_input(
                    f"{prefix}b",
                    min_value=1e-6,
                    value=2.0,
                    step=0.1,
                    key=f"{key_prefix}_b",
                )
        else:
            a = st.number_input(
                f"{param_name} a",
                min_value=1e-6,
                value=2.0,
                step=0.1,
                key=f"{key_prefix}_a",
            )
            b = st.number_input(
                f"{param_name} b",
                min_value=1e-6,
                value=2.0,
                step=0.1,
                key=f"{key_prefix}_b",
            )
        return {"a": float(a), "b": float(b)}

    if dist_name == "gamma":
        if layout == "columns":
            shape_col, scale_col = st.columns(2)
            with shape_col:
                a = st.number_input(
                    f"{prefix}shape",
                    min_value=1e-6,
                    value=2.0,
                    step=0.1,
                    key=f"{key_prefix}_a",
                )
            with scale_col:
                scale = st.number_input(
                    f"{prefix}scale",
                    min_value=1e-6,
                    value=float(_default_scale(default_val)),
                    step=0.01,
                    key=f"{key_prefix}_scale",
                )
        else:
            a = st.number_input(
                f"{param_name} shape",
                min_value=1e-6,
                value=2.0,
                step=0.1,
                key=f"{key_prefix}_a",
            )
            scale = st.number_input(
                f"{param_name} scale",
                min_value=1e-6,
                value=float(_default_scale(default_val)),
                step=0.01,
                key=f"{key_prefix}_scale",
            )
        return {"a": float(a), "scale": float(scale)}

    if dist_name == "expon":
        if layout == "columns":
            (scale_col,) = st.columns(1)
            with scale_col:
                scale = st.number_input(
                    f"{prefix}scale",
                    min_value=1e-6,
                    value=float(_default_scale(default_val)),
                    step=0.01,
                    key=f"{key_prefix}_scale",
                )
        else:
            scale = st.number_input(
                f"{param_name} scale",
                min_value=1e-6,
                value=float(_default_scale(default_val)),
                step=0.01,
                key=f"{key_prefix}_scale",
            )
        return {"scale": float(scale)}

    if dist_name == "uniform":
        if layout == "columns":
            loc_col, scale_col = st.columns(2)
            with loc_col:
                loc = st.number_input(
                    f"{prefix}loc",
                    value=0.0,
                    step=0.01,
                    key=f"{key_prefix}_loc",
                )
            with scale_col:
                scale = st.number_input(
                    f"{prefix}scale",
                    min_value=1e-6,
                    value=float(_default_scale(default_val)),
                    step=0.01,
                    key=f"{key_prefix}_scale",
                )
        else:
            loc = st.number_input(
                f"{param_name} loc",
                value=0.0,
                step=0.01,
                key=f"{key_prefix}_loc",
            )
            scale = st.number_input(
                f"{param_name} scale",
                min_value=1e-6,
                value=float(_default_scale(default_val)),
                step=0.01,
                key=f"{key_prefix}_scale",
            )
        return {"loc": float(loc), "scale": float(scale)}
    if dist_name == "constant":
        if layout == "columns":
            (value_col,) = st.columns(1)
            with value_col:
                value = st.number_input(
                    f"{prefix}value",
                    value=float(default_val),
                    step=0.01,
                    key=f"{key_prefix}_value",
                )
        else:
            value = st.number_input(
                f"{param_name} value",
                value=float(default_val),
                step=0.01,
                key=f"{key_prefix}_value",
            )
        return {"value": float(value)}

    st.warning(f"Unknown distribution {dist_name!r}; defaulting to norm.")
    return {"loc": float(default_val), "scale": float(_default_scale(default_val))}


@st.cache_data(ttl=2, show_spinner=False)
def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(ttl=2, show_spinner=False)
def _read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def _list_run_dirs(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir()]
    return sorted(dirs, key=lambda p: p.name, reverse=True)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_records_table(run_dir: Path) -> pd.DataFrame | None:
    csv_path = run_dir / "parameter_recovery_records.csv"
    parquet_path = run_dir / "parameter_recovery_records.parquet"
    if csv_path.exists():
        return _read_csv(str(csv_path))
    if parquet_path.exists():
        return _read_parquet(str(parquet_path))
    return None


def _load_metrics_table(run_dir: Path, records: pd.DataFrame | None) -> pd.DataFrame | None:
    metrics_path = run_dir / "parameter_recovery_metrics.csv"
    if metrics_path.exists():
        return _read_csv(str(metrics_path))
    if records is None:
        return None
    return compute_parameter_recovery_metrics(records)


def _merge_target_and_color(
    *,
    df: pd.DataFrame,
    target_param: str,
    color_param: str,
) -> pd.DataFrame:
    join_cols = [c for c in ("rep", "subject_id") if c in df.columns]
    if not join_cols:
        raise ValueError("Results table must include 'rep' or 'subject_id' columns.")

    target = df.loc[df["param"] == target_param, join_cols + ["true", "hat"]].copy()
    color = df.loc[df["param"] == color_param, join_cols + ["true", "hat"]].copy()
    if target.empty or color.empty:
        raise ValueError("Target or color parameter not found in results.")

    target.rename(columns={"true": "target_true", "hat": "target_hat"}, inplace=True)
    color.rename(columns={"true": "color_true", "hat": "color_hat"}, inplace=True)
    merged = target.merge(color, on=join_cols, how="inner")
    if merged.empty:
        raise ValueError("No matching rows after merging target and color parameters.")
    return merged


def _extract_value(df: pd.DataFrame, *, prefix: str, field: str) -> np.ndarray:
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
    raise ValueError(f"Unknown field: {field}")


def _render_results_view() -> None:
    st.header("Results")
    if px is None:
        st.error("Plotly is required for the Results view. Install with: pip install plotly")
        return

    auto_refresh = st.sidebar.checkbox("Auto-refresh results", value=True)
    if auto_refresh:
        autorefresh = getattr(st, "autorefresh", None)
        if callable(autorefresh):
            autorefresh(interval=5000, key="results_autorefresh")
        else:
            st.sidebar.caption("Auto-refresh is not available in this Streamlit version.")

    runs_root = st.text_input("Runs root directory", value="apps/parameter_recovery_gui/runs", key="results_root")
    run_dirs = _list_run_dirs(Path(runs_root))
    if not run_dirs:
        st.info("No runs found. Run a recovery first.")
        return

    run_labels = [p.name for p in run_dirs]
    selected_label = st.selectbox("Run directory", run_labels, index=0)
    run_dir = run_dirs[run_labels.index(selected_label)]

    manifest = _load_json(run_dir / "run_manifest.json")
    config_json = _load_json(run_dir / "parameter_recovery_config.json")
    col_manifest, col_config = st.columns(2)
    with col_manifest:
        with st.expander("Run manifest", expanded=False):
            st.json(manifest or {})
    with col_config:
        with st.expander("Recovery config", expanded=False):
            st.json(config_json or {})

    records = _load_records_table(run_dir)
    if records is None or records.empty:
        st.warning("No parameter recovery records found in this run directory.")
        return

    if "rep" not in records.columns:
        records = records.copy()
        records["rep"] = 0
        st.warning("No 'rep' column found; assuming a single replication.")

    reps = sorted(records["rep"].unique().tolist())
    params = sorted(records["param"].unique().tolist())

    st.subheader("Filters")
    rep_filter = st.multiselect("Replications", reps, default=reps)
    if not rep_filter:
        rep_filter = reps
    filtered = records.loc[records["rep"].isin(rep_filter)].copy()

    metrics = _load_metrics_table(run_dir, filtered)
    if metrics is not None and "rep" in metrics.columns:
        metrics = metrics.loc[metrics["rep"].isin(rep_filter)].copy()

    st.subheader("Metrics (per replication)")
    if metrics is not None:
        st.dataframe(metrics, use_container_width=True)
        summary = (
            metrics.groupby("param", sort=True)[["corr", "rmse", "mae", "bias", "median_abs_error"]]
            .mean()
            .reset_index()
        )
        st.caption("Mean metrics across replications.")
        st.dataframe(summary, use_container_width=True)
    else:
        st.info("No metrics table found.")

    st.subheader("True vs Hat Scatter")
    target_param = st.selectbox("Target parameter", params, index=0)
    color_mode = st.radio("Color by", ["rep", "parameter"], horizontal=True, index=0)
    rep_values = sorted(filtered["rep"].unique().tolist())
    if rep_values:
        rep_selected = st.selectbox("Replication", rep_values, index=0)
    else:
        rep_selected = None

    col_min, col_max = st.columns(2)
    with col_min:
        axis_min = st.number_input("Axis min", value=-10.0)
    with col_max:
        axis_max = st.number_input("Axis max", value=10.0)

    axis_min = float(axis_min)
    axis_max = float(axis_max)
    if axis_max <= axis_min:
        st.warning("Axis max must be greater than axis min. Adjusting range.")
        axis_min, axis_max = sorted([axis_min, axis_max])
        if axis_max == axis_min:
            axis_max = axis_min + 1.0

    color_param = None
    color_field = None
    if color_mode == "parameter":
        color_param_choices = [p for p in params if p != target_param]
        if not color_param_choices:
            st.warning("No other parameters available for color encoding.")
            return
        color_param = st.selectbox("Color parameter", color_param_choices, index=0)
        color_field = st.selectbox("Color value", ["true", "hat", "error", "abs_error"], index=0)

    rep_df = filtered if rep_selected is None else filtered.loc[filtered["rep"] == rep_selected].copy()
    if rep_selected is None:
        st.info("No rep column found; plotting all rows together.")
    else:
        st.caption(f"Showing replication {rep_selected}.")
    if color_mode == "parameter":
        merged = _merge_target_and_color(df=rep_df, target_param=target_param, color_param=color_param)
        merged["color_value"] = _extract_value(merged, prefix="color", field=color_field)
        plot_df = merged.dropna(subset=["target_true", "target_hat", "color_value"]).copy()
        rep_label = f"rep {rep_selected}" if rep_selected is not None else "all reps"
        title = f"{target_param}: true vs hat ({rep_label}, color = {color_param} {color_field})"
        fig = px.scatter(
            plot_df,
            x="target_true",
            y="target_hat",
            color="color_value",
            color_continuous_scale="Viridis",
            hover_data=[c for c in ("rep", "subject_id") if c in plot_df.columns],
            labels={"target_true": "True", "target_hat": "Hat", "color_value": f"{color_param} ({color_field})"},
            title=title,
        )
    else:
        plot_df = rep_df.loc[rep_df["param"] == target_param].dropna(subset=["true", "hat"]).copy()
        rep_label = f"rep {rep_selected}" if rep_selected is not None else "all reps"
        title = f"{target_param}: true vs hat ({rep_label})"
        fig = px.scatter(
            plot_df,
            x="true",
            y="hat",
            hover_data=[c for c in ("rep", "subject_id") if c in plot_df.columns],
            labels={"true": "True", "hat": "Hat"},
            title=title,
        )

    if plot_df.empty:
        return

    fig.add_shape(
        type="line",
        x0=axis_min,
        y0=axis_min,
        x1=axis_max,
        y1=axis_max,
        line=dict(color="rgba(200,200,200,0.8)", dash="dash"),
    )

    fig.update_xaxes(range=[axis_min, axis_max], constrain="domain")
    fig.update_yaxes(
        range=[axis_min, axis_max],
        scaleanchor="x",
        scaleratio=1,
        constrain="domain",
    )
    fig.update_layout(height=520)

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Per-rep correlation")
    if metrics is not None and not metrics.empty:
        corr_df = metrics.loc[metrics["param"] == target_param, ["corr"]].copy()
        if not corr_df.empty:
            fig_corr = px.histogram(
                corr_df,
                x="corr",
                nbins=20,
                range_x=[0.0, 1.0],
                labels={"corr": "Correlation"},
            )
            st.plotly_chart(fig_corr, use_container_width=True)


st.set_page_config(page_title="Parameter Recovery GUI", layout="wide")

st.title("Parameter Recovery GUI")
st.caption("Selections here generate YAML. Use the editors to override details.")

page = st.sidebar.radio("Mode", ["Run", "Results"], index=0)
if page == "Results":
    _render_results_view()
    st.stop()

registry = make_registry()
model_names = registry.models.names()
bandit_names = registry.bandits.names()
demo_names = registry.demonstrators.names()

col1, col2, col3 = st.columns(3)
with col1:
    true_model_name = st.selectbox("True model", model_names, index=0)
    fitted_model_name = st.selectbox("Fitted model", model_names, index=0)

with col2:
    n_subjects = int(st.number_input("Subjects", min_value=1, value=20, step=1))
    n_blocks = int(st.number_input("Blocks per subject per condition", min_value=1, value=1, step=1))
    n_trials = int(st.number_input("Trials per block", min_value=1, value=60, step=1))

with col3:
    conditions_selected = st.multiselect(
        "Conditions (cycled across blocks)",
        DEFAULT_CONDITIONS,
        default=[DEFAULT_CONDITIONS[0]],
    )
    if not conditions_selected:
        conditions_selected = [DEFAULT_CONDITIONS[0]]
        st.warning("At least one condition is required. Defaulting to 'A'.")
    baseline_condition = st.selectbox("Baseline condition", conditions_selected, index=0)
    is_social_choice = st.selectbox("Is social?", ["No", "Yes"], index=0)

st.caption("Total blocks per subject = (#conditions) × (blocks per condition).")

is_social = is_social_choice == "Yes"

true_model = registry.models[true_model_name]()
fitted_model = registry.models[fitted_model_name]()

true_is_social = isinstance(true_model, SocialComputationalModel)
fit_is_social = isinstance(fitted_model, SocialComputationalModel)

if is_social and not true_is_social:
    st.warning("True model is asocial but 'Is social' is set to Yes. The run may fail.")
if not is_social and (true_is_social or fit_is_social):
    st.warning("A selected model is social, but 'Is social' is set to No. The run may fail.")

prev_model = st.session_state.get("true_model_name")
if prev_model != true_model_name:
    for k in list(st.session_state.keys()):
        if str(k).startswith("true_param_"):
            st.session_state.pop(k, None)
    st.session_state["true_model_name"] = true_model_name

schema = getattr(true_model, "param_schema", None)
if schema is None:
    st.info("True model does not expose a param_schema; using defaults where possible.")

sampling_mode = st.selectbox("Sampling mode", ["fixed", "independent", "hierarchical"], index=0)
sampling_space = st.selectbox("Sampling space", ["param", "z"], index=0)

param_values: dict[str, float] = {}
sampling_individual: dict[str, dict[str, Any]] = {}
sampling_population: dict[str, dict[str, Any]] = {}
sampling_individual_sd: dict[str, float] = {}

if schema is not None:
    if sampling_mode == "fixed":
        st.subheader("True model parameters")
        for p in getattr(schema, "params", []):
            name = str(getattr(p, "name", "param"))
            default_val = float(getattr(p, "default", 0.0))
            bound = getattr(p, "bound", None)
            key = f"true_param_{name}"
            if key not in st.session_state:
                st.session_state[key] = default_val

            min_value = None
            max_value = None
            if bound is not None:
                try:
                    lo = float(getattr(bound, "lo"))
                    hi = float(getattr(bound, "hi"))
                    if math.isfinite(lo):
                        min_value = lo
                    if math.isfinite(hi):
                        max_value = hi
                except Exception:
                    min_value = None
                    max_value = None

            with st.container():
                st.markdown(f"**{name}**")
                if min_value is None and max_value is None:
                    val = st.number_input("Value", value=float(st.session_state[key]), step=0.01, key=key)
                else:
                    val = st.number_input(
                        "Value",
                        value=float(st.session_state[key]),
                        min_value=min_value,
                        max_value=max_value,
                        step=0.01,
                        key=key,
                    )
            param_values[name] = float(val)

    elif sampling_mode == "independent":
        st.subheader("Independent sampling distributions")
        for p in getattr(schema, "params", []):
            name = str(getattr(p, "name", "param"))
            default_val = float(getattr(p, "default", 0.0))
            dist_key = f"indiv_dist_{name}"
            if dist_key not in st.session_state:
                st.session_state[dist_key] = "norm"
            with st.container():
                st.markdown(f"**{name}**")
                dist_name = st.selectbox(
                    "Distribution",
                    DIST_FAMILIES,
                    index=DIST_FAMILIES.index(st.session_state[dist_key]),
                    key=dist_key,
                )
                args = _dist_args_inputs(
                    dist_name=dist_name,
                    param_name=name,
                    default_val=default_val,
                    key_prefix=f"indiv_{name}",
                    layout="columns",
                )
            sampling_individual[name] = {"name": dist_name, "args": args}

    elif sampling_mode == "hierarchical":
        st.subheader("Hierarchical population distributions")
        for p in getattr(schema, "params", []):
            name = str(getattr(p, "name", "param"))
            default_val = float(getattr(p, "default", 0.0))
            dist_key = f"pop_dist_{name}"
            if dist_key not in st.session_state:
                st.session_state[dist_key] = "norm"
            with st.container():
                st.markdown(f"**{name}**")
                dist_col, sd_col = st.columns([2, 1])
                with dist_col:
                    dist_name = st.selectbox(
                        "Population distribution",
                        DIST_FAMILIES,
                        index=DIST_FAMILIES.index(st.session_state[dist_key]),
                        key=dist_key,
                    )
                with sd_col:
                    sd = st.number_input(
                        "Subject SD",
                        min_value=1e-6,
                        value=float(_default_scale(default_val)),
                        step=0.01,
                        key=f"sd_{name}",
                    )
                args = _dist_args_inputs(
                    dist_name=dist_name,
                    param_name=name,
                    default_val=default_val,
                    key_prefix=f"pop_{name}",
                    layout="columns",
                )
            sampling_population[name] = {"name": dist_name, "args": args}
            sampling_individual_sd[name] = float(sd)
else:
    if sampling_mode != "fixed":
        st.warning("Sampling mode requires param_schema; switch to 'fixed' or edit YAML directly.")

bandit_name = st.selectbox("Bandit", bandit_names, index=0)
bandit_config: dict[str, Any] = {}

if bandit_name == "BernoulliBanditEnv":
    n_arms = int(st.number_input("Number of arms", min_value=2, value=2, step=1))
    probs: list[float] = []
    cols = st.columns(min(n_arms, 5))
    for i in range(n_arms):
        col = cols[i % len(cols)]
        with col:
            key = f"bandit_prob_{i}"
            if key not in st.session_state:
                st.session_state[key] = 1.0 / n_arms
            p = st.number_input(
                f"p[{i+1}]",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state[key]),
                step=0.01,
                key=key,
            )
            probs.append(float(p))

    total = sum(probs)
    if abs(total - 1.0) > 1e-3:
        st.warning(f"Probabilities sum to {total:.3f} (expected 1.0).")
    bandit_config = {"probs": probs}
else:
    st.info("This bandit has no dropdown-config yet. Edit YAML for custom settings if needed.")

if bandit_config.get("probs"):
    n_actions = len(bandit_config["probs"])
else:
    n_actions = 2

if is_social:
    demo_name = st.selectbox("Demonstrator", demo_names, index=0)
    demo_config: dict[str, Any] = {}

    if demo_name == "FixedSequenceDemonstrator":
        default_actions = _default_action_sequence(n_trials, n_actions)
        if (
            st.session_state.get("demo_actions_n_trials") != n_trials
            or st.session_state.get("demo_actions_n_actions") != n_actions
        ):
            st.session_state["demo_actions_text"] = default_actions
            st.session_state["demo_actions_n_trials"] = n_trials
            st.session_state["demo_actions_n_actions"] = n_actions

        actions_text = st.text_input(
            "Action sequence (comma-separated, 0-indexed)",
            value=st.session_state.get("demo_actions_text", default_actions),
            key="demo_actions_text",
        )
        try:
            actions = _parse_action_sequence(actions_text)
        except Exception as exc:
            st.warning(f"Invalid action sequence: {exc}")
            actions = []
        if not actions:
            st.warning("Action sequence is empty. This demonstrator will fail.")
        if len(actions) < n_trials:
            st.warning(
                f"Sequence length {len(actions)} < n_trials {n_trials}. "
                "FixedSequenceDemonstrator will raise an error."
            )
        if any(a < 0 or a >= n_actions for a in actions):
            st.warning(f"Actions must be in [0, {n_actions - 1}].")
        demo_config = {"actions": actions}

    elif demo_name == "NoisyBestArmDemonstrator":
        p_best = st.select_slider(
            "p_best",
            options=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            value=0.9,
        )
        demo_config = {"p_best": float(p_best)}

    elif demo_name == "RLDemonstrator":
        demo_model_name = st.selectbox("Demonstrator model", model_names, index=0)
        demo_model = registry.models[demo_model_name]()
        if isinstance(demo_model, SocialComputationalModel):
            st.warning("Demonstrator models are typically asocial; this one is social.")

        prev_demo_model = st.session_state.get("demo_model_name")
        if prev_demo_model != demo_model_name:
            for k in list(st.session_state.keys()):
                if str(k).startswith("demo_param_"):
                    st.session_state.pop(k, None)
            st.session_state["demo_model_name"] = demo_model_name

        demo_params: dict[str, float] = {}
        demo_schema = getattr(demo_model, "param_schema", None)
        if demo_schema is None:
            st.warning("Demonstrator model has no param_schema; params will be empty.")
        else:
            st.write("Demonstrator model parameters:")
            for p in getattr(demo_schema, "params", []):
                name = str(getattr(p, "name", "param"))
                default_val = float(getattr(p, "default", 0.0))
                bound = getattr(p, "bound", None)
                key = f"demo_param_{demo_model_name}_{name}"
                if key not in st.session_state:
                    st.session_state[key] = default_val

                min_value = None
                max_value = None
                if bound is not None:
                    try:
                        lo = float(getattr(bound, "lo"))
                        hi = float(getattr(bound, "hi"))
                        if math.isfinite(lo):
                            min_value = lo
                        if math.isfinite(hi):
                            max_value = hi
                    except Exception:
                        min_value = None
                        max_value = None

                if min_value is None and max_value is None:
                    val = st.number_input(
                        name,
                        value=float(st.session_state[key]),
                        step=0.01,
                        key=key,
                    )
                else:
                    val = st.number_input(
                        name,
                        value=float(st.session_state[key]),
                        min_value=min_value,
                        max_value=max_value,
                        step=0.01,
                        key=key,
                    )
                demo_params[name] = float(val)

        demo_config = {"model": demo_model_name, "params": demo_params}

    st.write("Demonstrator config:")
    st.code(yaml.safe_dump(demo_config, sort_keys=False).strip() or "{}", language="yaml")
else:
    demo_name = None
    demo_config = None

trial_spec_template: dict[str, Any] = {
    "self_outcome": {"kind": "VERIDICAL"},
    "available_actions": list(range(n_actions)),
}
if is_social:
    trial_spec_template["demo_outcome"] = {"kind": "VERIDICAL"}

base_block: dict[str, Any] = {
    "n_trials": n_trials,
    "bandit_type": bandit_name,
    "bandit_config": bandit_config,
    "trial_spec_template": trial_spec_template,
}

if is_social and demo_name and demo_config is not None:
    base_block["demonstrator_type"] = demo_name
    base_block["demonstrator_config"] = demo_config


def _build_blocks(*, n_blocks: int, conditions: list[str], base: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for cond in conditions:
        for rep in range(1, n_blocks + 1):
            block = dict(base)
            block["block_id"] = f"{cond}_b{rep}"
            block["condition"] = cond
            blocks.append(block)
    return blocks


blocks = _build_blocks(n_blocks=n_blocks, conditions=list(conditions_selected), base=base_block)

def _default_fixed_params(model_obj: Any) -> dict[str, float]:
    schema = getattr(model_obj, "param_schema", None)
    if schema is None:
        return {}
    return {str(k): float(v) for k, v in schema.defaults().items()}

if sampling_mode == "fixed":
    fixed_params = param_values if param_values else _default_fixed_params(true_model)
else:
    fixed_params = {}

output_root = st.text_input(
    "Output root directory",
    value="apps/parameter_recovery_gui/runs",
)

n_reps = int(st.number_input("Replications", min_value=1, value=20, step=1))
seed = int(st.number_input("Seed", min_value=0, value=0, step=1))

plan_dict = {
    "subjects": n_subjects,
    "subject_template": {
        "blocks": blocks,
    },
}

config_dict = {
    "plan_path": "__PLAN_PATH__",
    "n_reps": n_reps,
    "seed": seed,
    "sampling": {
        "mode": sampling_mode,
        "space": sampling_space,
        "fixed": fixed_params,
        "individual": sampling_individual,
        "population": sampling_population,
        "individual_sd": sampling_individual_sd,
        "clip_to_bounds": True,
    },
    "output": {
        "out_dir": output_root,
        "save_format": "csv",
        "save_config": True,
        "save_fit_diagnostics": True,
        "save_simulated_study": False,
    },
}

def _dump_yaml(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False)

if "plan_yaml" not in st.session_state:
    st.session_state.plan_yaml = _dump_yaml(plan_dict)
if "config_yaml" not in st.session_state:
    st.session_state.config_yaml = _dump_yaml(config_dict)

col_sync, col_note = st.columns([1, 3])
with col_sync:
    if st.button("Update YAML from dropdowns"):
        st.session_state.plan_yaml = _dump_yaml(plan_dict)
        st.session_state.config_yaml = _dump_yaml(config_dict)
with col_note:
    st.caption("YAML editors below are not auto-synced to avoid overwriting manual edits.")

plan_tab, config_tab = st.tabs(["Plan YAML", "Recovery Config YAML"])
with plan_tab:
    st.session_state.plan_yaml = st.text_area(
        "Study plan (YAML)",
        value=st.session_state.plan_yaml,
        height=320,
    )
with config_tab:
    st.session_state.config_yaml = st.text_area(
        "Parameter recovery config (YAML)",
        value=st.session_state.config_yaml,
        height=320,
    )

st.divider()

estimator_choice = st.selectbox(
    "Estimator",
    [
        "Auto (based on conditions)",
        "Transformed MLE (subjectwise)",
        "Box MLE (subjectwise)",
        "Within-subject shared+delta (transformed MLE)",
        "Bayesian hierarchical (Stan NUTS)",
    ],
    index=0,
)

bayes_hier = estimator_choice == "Bayesian hierarchical (Stan NUTS)"
bayes_required_priors: list[str] | None = None
bayes_adapter_error: str | None = None
bayes_estimator_args: dict[str, Any] = {}

if bayes_hier:
    st.subheader("Bayesian hierarchical settings")
    st.caption("Requires CmdStan + cmdstanpy; first run may compile Stan code.")
    if sampling_mode != "hierarchical":
        st.warning("Population recovery metrics require sampling.mode=hierarchical.")

    try:
        adapter = resolve_stan_adapter(fitted_model)
        bayes_required_priors = list(adapter.required_priors("hier"))
    except Exception as exc:
        bayes_adapter_error = str(exc)
        st.error(f"No Stan adapter available for {fitted_model_name!r}: {exc}")

    col1, col2, col3 = st.columns(3)
    with col1:
        chains = int(st.number_input("Chains", min_value=1, value=4, step=1, key="bayes_chains"))
        iter_warmup = int(st.number_input("Warmup iters", min_value=1, value=800, step=50, key="bayes_warmup"))
    with col2:
        iter_sampling = int(st.number_input("Sampling iters", min_value=1, value=1200, step=50, key="bayes_samples"))
        adapt_delta = float(
            st.number_input("Adapt delta", min_value=0.5, max_value=0.999, value=0.92, step=0.01, key="bayes_adapt")
        )
    with col3:
        max_treedepth = int(
            st.number_input("Max treedepth", min_value=5, max_value=20, value=12, step=1, key="bayes_treedepth")
        )
        show_progress = st.checkbox("Show CmdStan progress", value=False, key="bayes_show_progress")

    forbid_extra_priors = st.checkbox("Forbid extra priors", value=True, key="bayes_forbid_extra_priors")

    if bayes_required_priors:
        sig = tuple(bayes_required_priors)
        if st.session_state.get("hyper_priors_sig") != sig or st.session_state.get("hyper_priors_model") != fitted_model_name:
            st.session_state["hyper_priors_yaml"] = yaml.safe_dump(
                _default_hyper_priors(fitted_model, bayes_required_priors),
                sort_keys=False,
            )
            st.session_state["hyper_priors_sig"] = sig
            st.session_state["hyper_priors_model"] = fitted_model_name

        st.caption(f"Required hyper priors: {', '.join(bayes_required_priors)}")
        st.session_state["hyper_priors_yaml"] = st.text_area(
            "Hyper priors (YAML)",
            value=st.session_state.get("hyper_priors_yaml", ""),
            height=260,
        )
    else:
        st.info("Hyper priors could not be inferred because no Stan adapter was found.")

    bayes_estimator_args = {
        "chains": chains,
        "iter_warmup": iter_warmup,
        "iter_sampling": iter_sampling,
        "adapt_delta": adapt_delta,
        "max_treedepth": max_treedepth,
        "show_progress": show_progress,
        "forbid_extra_priors": forbid_extra_priors,
    }

if st.button("Run parameter recovery"):
    status_note = st.empty()
    progress_note = st.empty()
    progress_bar = st.progress(0)

    status_note.info("Preparing run...")
    with st.spinner("Running parameter recovery..."):
        try:
            status_note.info("Parsing YAML...")
            plan_raw = yaml.safe_load(st.session_state.plan_yaml)
            cfg_raw = yaml.safe_load(st.session_state.config_yaml)
        except Exception as exc:
            st.error(f"Failed to parse YAML: {exc}")
            st.stop()

        if not isinstance(plan_raw, dict):
            st.error("Plan YAML must be a mapping/object.")
            st.stop()
        if not isinstance(cfg_raw, dict):
            st.error("Config YAML must be a mapping/object.")
            st.stop()

        status_note.info("Writing run configuration...")
        plan_has_demo = _plan_mentions_demonstrator(plan_raw)
        if plan_has_demo and not is_social:
            st.warning("Plan includes a demonstrator, but 'Is social' is set to No. Using a social generator.")
        if not plan_has_demo and is_social:
            st.warning("Plan has no demonstrator, but 'Is social' is set to Yes. Using an asocial generator.")

        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root = Path(output_root) / "app_configs" / run_stamp
        run_root.mkdir(parents=True, exist_ok=True)

        plan_path = run_root / "study_plan.yaml"
        plan_path.write_text(yaml.safe_dump(plan_raw, sort_keys=False), encoding="utf-8")

        cfg_raw["plan_path"] = str(plan_path)
        cfg_text = yaml.safe_dump(cfg_raw, sort_keys=False)
        cfg_path = run_root / "recovery_config.yaml"
        cfg_path.write_text(cfg_text, encoding="utf-8")

        try:
            status_note.info("Loading recovery config...")
            config = load_parameter_recovery_config(cfg_path)
        except Exception as exc:
            st.error(f"Failed to load recovery config: {exc}")
            st.stop()

        try:
            status_note.info("Inferring study conditions...")
            conditions = infer_load_conditions(plan_path)
        except Exception as exc:
            st.error(f"Failed to infer conditions: {exc}")
            st.stop()

        baseline = baseline_condition if baseline_condition in conditions else conditions[0]
        if baseline_condition not in conditions:
            st.warning(f"Baseline condition {baseline_condition!r} not found in plan; using {baseline!r}.")

        bayes_hyper_priors_raw: dict[str, Any] | None = None

        if estimator_choice == "Auto (based on conditions)":
            use_within_subject = len(conditions) > 1
            if use_within_subject:
                estimator = WithinSubjectSharedDeltaTransformedMLEEstimator(
                    base_model=fitted_model,
                    baseline_condition=baseline,
                    conditions=conditions,
                )
            else:
                estimator = TransformedMLESubjectwiseEstimator(model=fitted_model)
        elif estimator_choice == "Transformed MLE (subjectwise)":
            estimator = TransformedMLESubjectwiseEstimator(model=fitted_model)
        elif estimator_choice == "Box MLE (subjectwise)":
            estimator = BoxMLESubjectwiseEstimator(model=fitted_model)
        elif estimator_choice == "Bayesian hierarchical (Stan NUTS)":
            if bayes_adapter_error:
                st.error(f"Cannot run Bayesian hierarchical estimator: {bayes_adapter_error}")
                st.stop()
            try:
                hyper_priors_raw = yaml.safe_load(st.session_state.get("hyper_priors_yaml", "")) or {}
            except Exception as exc:
                st.error(f"Failed to parse hyper priors YAML: {exc}")
                st.stop()
            if not isinstance(hyper_priors_raw, dict):
                st.error("Hyper priors YAML must be a mapping/object.")
                st.stop()
            if bayes_required_priors:
                missing = [k for k in bayes_required_priors if k not in hyper_priors_raw]
                if missing:
                    st.error(f"Missing required hyper priors: {missing}")
                    st.stop()
            bayes_hyper_priors_raw = hyper_priors_raw
            estimator = StanHierarchicalNUTSEstimator(
                model=fitted_model,
                hyper_priors=hyper_priors_raw,
                chains=int(bayes_estimator_args.get("chains", 4)),
                iter_warmup=int(bayes_estimator_args.get("iter_warmup", 800)),
                iter_sampling=int(bayes_estimator_args.get("iter_sampling", 1200)),
                adapt_delta=float(bayes_estimator_args.get("adapt_delta", 0.92)),
                max_treedepth=int(bayes_estimator_args.get("max_treedepth", 12)),
                forbid_extra_priors=bool(bayes_estimator_args.get("forbid_extra_priors", True)),
                show_progress=bool(bayes_estimator_args.get("show_progress", False)),
            )
        elif estimator_choice == "Within-subject shared+delta (transformed MLE)":
            if len(conditions) < 2:
                st.warning("Within-subject estimator selected, but only one condition detected.")
            estimator = WithinSubjectSharedDeltaTransformedMLEEstimator(
                base_model=fitted_model,
                baseline_condition=baseline,
                conditions=conditions,
            )
        else:
            st.error(f"Unknown estimator choice: {estimator_choice!r}")
            st.stop()

        if bayes_hier and bayes_hyper_priors_raw is not None:
            priors_path = run_root / "hyper_priors.yaml"
            priors_path.write_text(
                yaml.safe_dump(bayes_hyper_priors_raw, sort_keys=False),
                encoding="utf-8",
            )
            est_cfg = {"type": "stan_hierarchical_nuts", **bayes_estimator_args}
            (run_root / "estimator_config.yaml").write_text(
                yaml.safe_dump(est_cfg, sort_keys=False),
                encoding="utf-8",
            )

        is_social_run = plan_has_demo
        generator = EventLogSocialPreChoiceGenerator() if is_social_run else EventLogAsocialGenerator()

        try:
            status_note.info("Running parameter recovery...")

            def _progress_callback(done: int, total: int) -> None:
                denom = max(total, 1)
                pct = int(100 * done / denom)
                progress_bar.progress(pct)
                progress_note.caption(f"Replication {done}/{total}")

            out = run_parameter_recovery(
                config=config,
                generator=generator,
                model=true_model,
                estimator=estimator,
                progress_callback=_progress_callback,
            )
        except Exception as exc:
            st.error(f"Run failed: {exc}")
            st.stop()

    progress_bar.progress(100)
    status_note.success("Run complete.")
    st.write(f"Output directory: `{out.out_dir}`")
    st.subheader("Metrics (head)")
    st.dataframe(out.metrics.head(20))
    if out.population_metrics is not None:
        st.subheader("Population metrics (head)")
        st.dataframe(out.population_metrics.head(20))
