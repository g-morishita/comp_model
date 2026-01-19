from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

try:
    import dash
    from dash import dcc, html, Input, Output
except Exception as e:  # noqa: BLE001
    dash = None  # type: ignore[assignment]
    dcc = None   # type: ignore[assignment]
    html = None  # type: ignore[assignment]
    Input = None # type: ignore[assignment]
    Output = None# type: ignore[assignment]
    _dash_import_error = e
else:
    _dash_import_error = None

import plotly.graph_objects as go


ColorMode = Literal["highlight_true_range", "true", "hat", "error", "abs_error"]


@dataclass(frozen=True, slots=True)
class DashboardSpec:
    """
    Dashboard settings.
    """
    host: str = "127.0.0.1"
    port: int = 8050
    debug: bool = False


def _require_columns(df: pd.DataFrame) -> None:
    needed = {"param", "true", "hat", "rep", "subject_id"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in df: {sorted(missing)}")


def _make_figure(
    g: pd.DataFrame,
    *,
    param: str,
    color_mode: ColorMode,
    range_lo: float,
    range_hi: float,
) -> go.Figure:
    """
    Build a plotly Figure for one param.
    """
    x = g["true"].to_numpy(dtype=float)
    y = g["hat"].to_numpy(dtype=float)

    # compute fields
    err = y - x
    abs_err = np.abs(err)

    # color logic
    if color_mode == "highlight_true_range":
        in_range = (x >= range_lo) & (x <= range_hi)
        # red for in-range, light gray otherwise
        colors = np.where(in_range, "red", "lightgray")
        marker = dict(color=colors, size=7, opacity=0.85)
        showscale = False
        colorbar = None
        c = None

    else:
        # continuous color scale
        if color_mode == "true":
            c = x
            ctitle = "True"
        elif color_mode == "hat":
            c = y
            ctitle = "Estimated"
        elif color_mode == "error":
            c = err
            ctitle = "Error (hat - true)"
        elif color_mode == "abs_error":
            c = abs_err
            ctitle = "Abs error |hat-true|"
        else:
            raise ValueError(f"Unknown color_mode: {color_mode}")

        marker = dict(
            color=c,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title=ctitle),
            size=7,
            opacity=0.85,
        )
        showscale = True
        colorbar = ctitle

    # identity line extents
    lo = float(min(np.nanmin(x), np.nanmin(y)))
    hi = float(max(np.nanmax(x), np.nanmax(y)))
    pad = 0.02 * (hi - lo + 1e-12)
    lo -= pad
    hi += pad

    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="markers",
            marker=marker,
            customdata=np.stack(
                [
                    g["subject_id"].astype(str).to_numpy(),
                    g["rep"].astype(int).to_numpy(),
                    x,
                    y,
                    err,
                    abs_err,
                ],
                axis=1,
            ),
            hovertemplate=(
                "subject=%{customdata[0]}<br>"
                "rep=%{customdata[1]}<br>"
                "true=%{customdata[2]:.4g}<br>"
                "hat=%{customdata[3]:.4g}<br>"
                "error=%{customdata[4]:.4g}<br>"
                "abs_error=%{customdata[5]:.4g}<extra></extra>"
            ),
            name="data",
        )
    )

    # identity line
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="identity",
        )
    )

    fig.update_layout(
        title=f"Parameter recovery: {param}",
        xaxis_title="True",
        yaxis_title="Estimated",
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
    )

    # lock aspect ratio 1:1 for easy visual bias detection
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    # show selected true-range in title when highlight mode
    if color_mode == "highlight_true_range":
        fig.update_layout(
            title=f"Parameter recovery: {param} (highlight true in [{range_lo:.3g}, {range_hi:.3g}])"
        )

    return fig


def run_parameter_recovery_dashboard(
    df: pd.DataFrame,
    *,
    spec: DashboardSpec = DashboardSpec(),
    initial_param: str | None = None,
    initial_color_mode: ColorMode = "highlight_true_range",
) -> None:
    """
    Launch an interactive Dash dashboard for parameter recovery.

    Install requirements:
      pip install dash plotly

    df must have columns: param, true, hat, rep, subject_id
    """
    if dash is None:
        raise ImportError(
            "Dash is required for the interactive dashboard. "
            "Install: pip install dash plotly"
        ) from _dash_import_error

    _require_columns(df)

    # clean + drop nans
    df = df.dropna(subset=["param", "true", "hat"]).copy()
    df["param"] = df["param"].astype(str)

    params = sorted(df["param"].unique().tolist())
    if not params:
        raise ValueError("No parameters found in df['param'].")

    if initial_param is None or initial_param not in params:
        initial_param = params[0]

    # slider range based on selected param
    g0 = df[df["param"] == initial_param]
    tmin = float(g0["true"].min())
    tmax = float(g0["true"].max())
    if not np.isfinite(tmin) or not np.isfinite(tmax):
        tmin, tmax = 0.0, 1.0

    # initialize slider to middle-ish range (or full range if degenerate)
    if tmax > tmin:
        lo0 = tmin + 0.3 * (tmax - tmin)
        hi0 = tmin + 0.7 * (tmax - tmin)
    else:
        lo0, hi0 = tmin, tmax

    app = dash.Dash(__name__)

    app.layout = html.Div(
        [
            html.H3("Parameter recovery dashboard"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Parameter"),
                            dcc.Dropdown(
                                id="param",
                                options=[{"label": p, "value": p} for p in params],
                                value=initial_param,
                                clearable=False,
                            ),
                        ],
                        style={"width": "30%", "display": "inline-block", "verticalAlign": "top"},
                    ),
                    html.Div(
                        [
                            html.Label("Color mode"),
                            dcc.Dropdown(
                                id="color_mode",
                                options=[
                                    {"label": "Highlight TRUE range (red)", "value": "highlight_true_range"},
                                    {"label": "Color by TRUE (continuous)", "value": "true"},
                                    {"label": "Color by HAT (continuous)", "value": "hat"},
                                    {"label": "Color by ERROR (hat-true)", "value": "error"},
                                    {"label": "Color by ABS_ERROR", "value": "abs_error"},
                                ],
                                value=initial_color_mode,
                                clearable=False,
                            ),
                        ],
                        style={"width": "35%", "display": "inline-block", "marginLeft": "2%", "verticalAlign": "top"},
                    ),
                ],
                style={"marginBottom": "12px"},
            ),
            html.Div(
                [
                    html.Label("Highlight TRUE range (adjustable)"),
                    dcc.RangeSlider(
                        id="true_range",
                        min=tmin,
                        max=tmax,
                        step=(tmax - tmin) / 200 if tmax > tmin else 0.01,
                        value=[lo0, hi0],
                        tooltip={"placement": "bottom", "always_visible": True},
                        allowCross=False,
                    ),
                ],
                style={"marginBottom": "18px"},
            ),
            dcc.Graph(id="fig", style={"height": "75vh"}),
            html.Div(
                "Tip: use box select to zoom; hover points to see rep/subject/error.",
                style={"marginTop": "8px", "color": "#555"},
            ),
        ],
        style={"padding": "14px"},
    )

    @app.callback(
        Output("true_range", "min"),
        Output("true_range", "max"),
        Output("true_range", "step"),
        Output("true_range", "value"),
        Input("param", "value"),
    )
    def _update_slider_for_param(param: str):
        g = df[df["param"] == param]
        tmin = float(g["true"].min())
        tmax = float(g["true"].max())
        if not np.isfinite(tmin) or not np.isfinite(tmax):
            tmin, tmax = 0.0, 1.0
        if tmax > tmin:
            lo = tmin + 0.3 * (tmax - tmin)
            hi = tmin + 0.7 * (tmax - tmin)
            step = (tmax - tmin) / 200
        else:
            lo, hi = tmin, tmax
            step = 0.01
        return tmin, tmax, step, [lo, hi]

    @app.callback(
        Output("fig", "figure"),
        Input("param", "value"),
        Input("color_mode", "value"),
        Input("true_range", "value"),
    )
    def _update_fig(param: str, color_mode: str, true_range: list[float]):
        g = df[df["param"] == param]
        lo, hi = float(true_range[0]), float(true_range[1])
        return _make_figure(g, param=param, color_mode=color_mode, range_lo=lo, range_hi=hi)

    app.run_server(host=spec.host, port=spec.port, debug=spec.debug)
