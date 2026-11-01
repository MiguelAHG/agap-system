"""
charts.py

Plotly figure builders. They only draw; all numbers come precomputed from model.py. Maps use the MapLibre traces
(go.Choroplethmap / go.Scattermap) when the installed Plotly has them, and fall back to the older Mapbox traces otherwise.
No Mapbox token is needed (OpenStreetMap tiles).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go

from . import model as M

FORECAST_COLOR = "#e8710a"
FORECAST_LINE = "#8f3a00"          # darker orange for the point forecast line
ACTUAL_COLOR = "#1f3b57"
GRID = "#e6e6e6"
RAIN_FILL = "rgba(120, 170, 220, 0.16)"
CHART_CONFIG = {"displayModeBar": False}

SLIDER_X, SLIDER_LEN = 0.14, 0.84                  # horizontal extent of the map slider (paper units)
CLUSTER_LINE = dict(width=2, color="#ff8a8a")        # light red outline between clusters

_NEW_MAPS = hasattr(go, "Choroplethmap")
_CHORO = go.Choroplethmap if _NEW_MAPS else go.Choroplethmapbox
_SCATTER = go.Scattermap if _NEW_MAPS else go.Scattermapbox
_MAP_KEY = "map" if _NEW_MAPS else "mapbox"


def _base_layout(fig: go.Figure, height: int, **extra) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", **extra)
    return fig


def _shade(rank: int, count: int) -> str:
    """Grey for an interval level; rank 0 is the widest level (lightest)."""
    t = 0 if count <= 1 else rank / (count - 1)
    v = int(round(184 - t * (184 - 68)))
    return f"rgb({v},{v},{v})"


# ----------------------------------------------------------------------------------------------
# Bar chart: forecasts for all horizons with labelled intervals
# ----------------------------------------------------------------------------------------------

def fig_forecast_bars(bars: pd.DataFrame, labels: pd.DataFrame, level_keys: list, y_title: str) -> go.Figure:
    n = len(bars)
    pos = list(range(n))
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=pos, y=bars["pred"], width=0.34, marker=dict(color=FORECAST_COLOR), name="Point forecast",
        customdata=[M.fmt_span(t) for t in bars["target_week"]],
        hovertemplate="<b>Week of %{customdata}</b><br>Point forecast: %{y:.1f}<extra></extra>",
    ))

    count = len(level_keys)
    for rank, k in enumerate(reversed(level_keys)):                      # widest first, so thicker lines sit on top
        xs, ys = [], []
        for p, lo, hi in zip(pos, bars[f"lo{k}"], bars[f"hi{k}"]):
            xs += [p, p, None]
            ys += [lo, hi, None]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=f"{k}% interval",
            line=dict(color=_shade(rank, count), width=2 + 4.5 * rank), hoverinfo="skip",
        ))

    offset = 0.21
    point = labels[labels["kind"] == "pred"]
    bounds = labels[labels["kind"] != "pred"]
    fig.add_trace(go.Scatter(
        x=[p + offset for p in point["horizon"].map({h: i for i, h in enumerate(bars["horizon"])})],
        y=point["label_y"], text=[f"<b>{t}</b>" for t in point["text"]], mode="text", textposition="middle right",
        textfont=dict(size=13, color="#a64f00"), hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[p + offset for p in bounds["horizon"].map({h: i for i, h in enumerate(bars["horizon"])})],
        y=bounds["label_y"], text=bounds["text"], mode="text", textposition="middle right",
        textfont=dict(size=11, color="#444444"), hoverinfo="skip", showlegend=False,
    ))

    top = max(labels["label_y"].max(), max(bars[f"hi{k}"].max() for k in level_keys)) * 1.06
    bottom = min(0.0, labels["label_y"].min() - 0.02 * top)
    ticks = [t for t in M.nice_ticks(top) if t >= 0]
    fig.update_xaxes(
        tickmode="array", tickvals=pos, showgrid=False, range=[-0.6, n - 1 + 0.85],
        ticktext=[f"<b>{M.horizon_label(int(h))}</b><br>{M.fmt_span(t)}"
                  for h, t in zip(bars["horizon"], bars["target_week"])],
    )
    fig.update_yaxes(range=[bottom, top], tickvals=ticks, gridcolor=GRID, title=y_title, zeroline=False)
    return _base_layout(fig, 470, bargap=0.2, legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0))


# ----------------------------------------------------------------------------------------------
# Line chart: actual cases vs forecasts with nested intervals
# ----------------------------------------------------------------------------------------------

def _add_rainy_shading(fig: go.Figure, first_year: int, last_year: int, label: bool = True) -> None:
    """Shade June to November of each year and label every season at its middle, near the bottom of the plot."""
    for year in range(first_year, last_year + 1):
        fig.add_shape(type="rect", x0=f"{year}-06-01", x1=f"{year}-12-01", y0=0, y1=1, xref="x", yref="paper",
                      fillcolor=RAIN_FILL, line_width=0, layer="below")
        if label:
            fig.add_annotation(x=f"{year}-09-01", y=0.01, xref="x", yref="paper", text="Rainy season", showarrow=False,
                               xanchor="center", yanchor="bottom", font=dict(size=11, color="steelblue"))


def fig_timeseries(ts: dict, horizon: int, y_title: str, show_forecast: bool = True, show_bands: bool = True,
                   show_rain: bool = True) -> go.Figure:
    fig = go.Figure()
    weeks = ts["weeks"]
    if show_rain:
        _add_rainy_shading(fig, pd.Timestamp(weeks.min()).year, pd.Timestamp(weeks.max()).year)

    sf = ts["show_forecast"]
    fx = weeks[sf]
    if show_bands and show_forecast:
        bands = ts["bands"]
        alphas = np.linspace(0.12, 0.42, len(bands))[::-1]               # widest level is the lightest
        for band, alpha in zip(reversed(bands), reversed(list(alphas))):  # draw widest first
            fig.add_trace(go.Scatter(
                x=np.concatenate([fx, fx[::-1]]), y=np.concatenate([band["hi"][sf], band["lo"][sf][::-1]]),
                fill="toself", fillcolor=f"rgba(232,113,10,{alpha:.2f})", line=dict(width=0),
                name=f"{band['label']} interval", hoverinfo="skip",
            ))
    if show_forecast:
        fig.add_trace(go.Scatter(
            x=fx, y=ts["pred"][sf], mode="lines", name="Point forecast", line=dict(color=FORECAST_LINE, width=3.5, dash="dash"),
            hovertemplate="Point forecast: %{y:.1f}<extra></extra>",
        ))
    if ts["actual"] is not None:
        sa = ts["show_actual"]
        fig.add_trace(go.Scatter(
            x=weeks[sa], y=ts["actual"][sa], mode="lines", name="Actual cases", line=dict(color=ACTUAL_COLOR, width=2),
            hovertemplate="Actual: %{y:.0f}<extra></extra>",
        ))

    origin, predicted = ts["origin"], ts["predicted_week"]
    for when, text, color, anchor in ((origin, f"Current week<br>{origin:%b %d}", "#c0392b", "right"),
                                      (predicted, f"Week forecasted<br>{predicted:%b %d}", "#1e8449", "left")):
        stamp = pd.Timestamp(when).strftime("%Y-%m-%d")
        fig.add_shape(type="line", x0=stamp, x1=stamp, y0=0, y1=1, yref="paper", line=dict(color=color, dash="dot", width=2))
        fig.add_annotation(x=stamp, y=0.99, yref="paper", text=text, showarrow=False, xanchor=anchor,
                           font=dict(color=color, size=12), bgcolor="rgba(255,255,255,0.75)")

    full0, full1 = ts["full_range"]
    w0, w1 = ts["window"]
    fig.update_xaxes(
        type="date", range=[w0.strftime("%Y-%m-%d"), w1.strftime("%Y-%m-%d")], showgrid=False, title="Week being forecasted",
        rangeslider=dict(visible=True, thickness=0.07, range=[full0.strftime("%Y-%m-%d"), full1.strftime("%Y-%m-%d")]),
        rangeselector=dict(buttons=[dict(count=1, label="1 year", step="year", stepmode="backward"),
                                    dict(step="all", label="All")]),
    )
    fig.update_yaxes(range=[0, ts["y_max"]], gridcolor=GRID, title=y_title, zeroline=False)
    return _base_layout(fig, 520, hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))


# ----------------------------------------------------------------------------------------------
# Historical trends
# ----------------------------------------------------------------------------------------------

def fig_trend(series: pd.Series, y_title: str, decimals: int, show_rain: bool, title: str = "",
              x_title: str = "Week") -> go.Figure:
    fig = go.Figure()
    if show_rain and len(series):
        _add_rainy_shading(fig, series.index.min().year, series.index.max().year, label=False)
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values, mode="lines", connectgaps=False, name=y_title,
        line=dict(color=ACTUAL_COLOR, width=1.8),
        hovertemplate=f"%{{x|%b %d, %Y}}<br>%{{y:,.{decimals}f}}<extra></extra>",
    ))
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.07), showgrid=False, title=x_title,
        rangeselector=dict(buttons=[
            dict(count=1, label="1 year", step="year", stepmode="backward"),
            dict(count=3, label="3 years", step="year", stepmode="backward"),
            dict(count=5, label="5 years", step="year", stepmode="backward"),
            dict(step="all", label="All"),
        ]),
    )
    fig.update_yaxes(gridcolor=GRID, title=y_title)
    _base_layout(fig, 480, showlegend=False)
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=16)) if title else None, margin=dict(l=10, r=10, t=60 if title else 30, b=10))
    return fig


MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def fig_seasonal(by_year: pd.DataFrame, mean: pd.Series, y_title: str, decimals: int, title: str = "",
                 x_title: str = "Month") -> go.Figure:
    fig = go.Figure()
    for i, year in enumerate(by_year.columns):                           # one gray legend entry for all individual years
        fig.add_trace(go.Scatter(
            x=by_year.index, y=by_year[year], mode="lines", name="Data from individual years", legendgroup="years",
            showlegend=(i == 0), line=dict(width=1, color="rgba(120,120,120,0.45)"),
            hovertemplate=f"{year}: %{{y:,.{decimals}f}}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=mean.index, y=mean.values, mode="lines+markers", name="Average over all years",
        line=dict(width=3.5, color=FORECAST_COLOR), hovertemplate=f"Average: %{{y:,.{decimals}f}}<extra></extra>",
    ))
    fig.update_xaxes(tickmode="array", tickvals=list(range(1, 13)), ticktext=MONTHS, showgrid=False, title=x_title)
    fig.update_yaxes(gridcolor=GRID, title=y_title)
    _base_layout(fig, 480, hovermode="x unified", showlegend=True,
                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    fig.update_layout(title=dict(text=title, x=0, y=0.97, yanchor="top", font=dict(size=16)) if title else None,
                      margin=dict(l=10, r=10, t=110 if title else 40, b=10))
    return fig


# ----------------------------------------------------------------------------------------------
# Maps
# ----------------------------------------------------------------------------------------------

_PLACEHOLDER = re.compile(r"\{(\d+)(:[^}]*)?\}")


def _hover_template(lines: list) -> str:
    rows = []
    for label, template in lines:
        value = _PLACEHOLDER.sub(lambda m: f"%{{customdata[{m.group(1)}]{m.group(2) or ''}}}", template)
        rows.append(f"{label}: {value}")
    return "<b>Cluster %{location}</b><br>" + "<br>".join(rows) + "<extra></extra>"


def _map_layout(fig: go.Figure, extras: dict, height: int, zoom: float = 11.0, **extra) -> go.Figure:
    center = extras["map_center"]
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0),
        **{_MAP_KEY: dict(style="open-street-map", center=dict(lat=center["lat"], lon=center["lon"]), zoom=zoom)},
        **extra,
    )
    return fig


def _marker_labels(units: list, extras: dict, texts: list) -> go.Scattermap:
    lons = [extras["centroids"][u][0] for u in units]
    lats = [extras["centroids"][u][1] for u in units]
    return _SCATTER(lon=lons, lat=lats, mode="markers+text", marker=dict(size=46, color="white", opacity=0.75, allowoverlap=True),
                    text=texts, textfont=dict(size=12, color="black"), hoverinfo="skip", showlegend=False)


def _week_box(noun: str, week) -> dict:
    """Box at the top left of the map: 'Inference week: 2025-07-04 - 2025-07-10' (week start to its 6th day)."""
    start = pd.Timestamp(week)
    text = f"{noun}: {start:%Y-%m-%d} - {start + pd.Timedelta(days=6):%Y-%m-%d}"
    return dict(x=0.01, y=0.99, xref="paper", yref="paper", xanchor="left", yanchor="top", showarrow=False, text=f"<b>{text}</b>",
                font=dict(size=14, color="#222222"), bgcolor="rgba(255,255,255,0.95)", bordercolor="#999999",
                borderwidth=1, borderpad=6, align="left")


def fig_map_animated(m: M.MapFrames, clusters_geojson: dict, extras: dict, height: int = 640, frame_ms: int = 450) -> go.Figure:
    """
    Cluster map with a built-in slider and Play button. All frames are sent to the browser once, so moving the slider or
    playing never triggers a Streamlit rerun.
    """
    i0 = m.initial_index
    choropleth = _CHORO(
        geojson=clusters_geojson, locations=m.units, z=m.z[i0], zmin=m.zmin, zmax=m.zmax, colorscale=m.colorscale,
        marker=dict(opacity=0.65, line=CLUSTER_LINE), customdata=m.custom[i0],
        hovertemplate=_hover_template(m.hover_lines),
        colorbar=dict(title=dict(text=m.colorbar_title), thickness=14, len=0.75),
    )
    fig = go.Figure(data=[choropleth, _marker_labels(m.units, extras, m.text[i0])])
    _map_layout(fig, extras, height)
    fig.update_layout(title=dict(text=m.title, x=0.01, y=0.985, font=dict(size=16)), margin=dict(l=0, r=0, t=48, b=0))

    if m.animated:
        labels = [pd.Timestamp(w).strftime("%Y-%m-%d") for w in m.weeks]
        fig.update_layout(annotations=[_week_box(m.week_noun, m.weeks[i0])])
        fig.frames = [
            go.Frame(name=labels[i], traces=[0, 1],
                     data=[_CHORO(z=m.z[i], customdata=m.custom[i]), _SCATTER(text=m.text[i])],
                     layout=dict(annotations=[_week_box(m.week_noun, m.weeks[i])]))
            for i in range(len(labels))
        ]
        # Year labels are drawn as the tick labels of a hidden axis just under the map, with the same x extent and equal
        # spacing as the slider steps (slider step labels are unreliable with hundreds of steps).
        years = [pd.Timestamp(w).year for w in m.weeks]
        first_of_year = [i for i, y in enumerate(years) if i == 0 or y != years[i - 1]]
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", opacity=0, hoverinfo="skip", showlegend=False))
        fig.update_layout(
            xaxis=dict(domain=[SLIDER_X, SLIDER_X + SLIDER_LEN], range=[0, max(len(years) - 1, 1)], anchor="y",
                       tickmode="array", tickvals=first_of_year, ticktext=[f"{years[i]}<br><span style='color:#888888'>|</span>" for i in first_of_year],
                       showgrid=False, zeroline=False, showline=False, ticks="",
                       tickfont=dict(size=12, color="#333333"), fixedrange=True),
            yaxis=dict(domain=[0, 0.001], range=[0, 1], visible=False, fixedrange=True),
        )
        step_args = dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))
        fig.update_layout(
            height=height + 150, margin=dict(l=0, r=0, t=48, b=140),
            updatemenus=[dict(
                type="buttons", direction="left", x=0.0, y=0, xanchor="left", yanchor="top", pad=dict(t=52, r=10),
                showactive=False,
                buttons=[
                    dict(label="Play", method="animate",
                         args=[None, dict(frame=dict(duration=frame_ms, redraw=True), transition=dict(duration=0),
                                          fromcurrent=False, mode="immediate")]),
                    dict(label="Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False), transition=dict(duration=0),
                                            mode="immediate")]),
                ],
            )],
            sliders=[dict(
                active=i0, x=SLIDER_X, y=0, len=SLIDER_LEN, xanchor="left", yanchor="top", pad=dict(t=52, b=0, l=0, r=0),
                currentvalue=dict(visible=False),
                ticklen=9, minorticklen=3, tickwidth=1, tickcolor="#888888", font=dict(size=12, color="#333333"),
                steps=[dict(method="animate", label=" ", args=[[lab], step_args]) for lab in labels],
            )],
        )
    return fig


def fig_cluster_guide(barangays_geojson: dict, brgy_table: pd.DataFrame, extras: dict, height: int = 640) -> go.Figure:
    clusters = sorted(brgy_table["cluster"].unique())
    palette = pc.qualitative.Dark24
    colors = {c: palette[i % len(palette)] for i, c in enumerate(clusters)}
    features = barangays_geojson["features"]
    fig = go.Figure()

    for i, c in enumerate(clusters):
        subset = brgy_table[brgy_table["cluster"] == c]
        fig.add_trace(_CHORO(
            geojson={"type": "FeatureCollection", "features": [f for f in features if f["properties"]["cluster"] == c]},
            locations=subset["barangay"], z=[0] * len(subset), zmin=0, zmax=1, showscale=False,
            colorscale=[[0, colors[c]], [1, colors[c]]], marker=dict(opacity=0.6, line=dict(width=0.8, color="white")),
            customdata=np.column_stack([subset["cluster"], subset["district"].astype(object).where(subset["district"].notna(), "n/a")]),
            hovertemplate="<b>%{location}</b><br>Cluster %{customdata[0]}<br>District %{customdata[1]}<extra></extra>",
            showlegend=False,
        ))
    for i, c in enumerate(clusters):                                     # legend entries
        fig.add_trace(_SCATTER(lat=[None], lon=[None], mode="markers", marker=dict(size=12, color=colors[c]),
                               name=f"Cluster {c}", legendgroup="clusters",
                               legendgrouptitle_text="Clusters" if i == 0 else None, hoverinfo="skip"))

    district_colors = ["#c0392b", "#2471a3", "#1e8449", "#d68910", "#7d3c98", "#6e4b3a", "#17202a", "#117a65"]
    for i, (name, line) in enumerate(sorted(extras["district_lines"].items())):
        color = district_colors[i % len(district_colors)]
        fig.add_trace(_SCATTER(lon=line["lon"], lat=line["lat"], mode="lines", line=dict(color=color, width=6),
                               name=f"District {name}", legendgroup="districts",
                               legendgrouptitle_text="Districts" if i == 0 else None,
                               visible="legendonly", hoverinfo="skip"))
        fig.add_trace(_SCATTER(lon=[line["centroid"][0]], lat=[line["centroid"][1]], mode="markers+text",
                               marker=dict(size=30, color="white", opacity=0.85, allowoverlap=True), text=[f"D{name}"],
                               textfont=dict(size=14, color=color), legendgroup="districts", visible="legendonly",
                               hoverinfo="skip", showlegend=False))
    _map_layout(fig, extras, height, hoverlabel=dict(font_size=15), legend=dict(bgcolor="rgba(255,255,255,0.85)"))
    return fig


def fig_performance_map(rmse_by_cluster: pd.Series, clusters_geojson: dict, extras: dict, height: int = 520) -> go.Figure:
    values = rmse_by_cluster.dropna()
    units = list(values.index)
    texts = [f"{u}<br>{v:.2f}" for u, v in values.items()]
    choropleth = _CHORO(
        geojson=clusters_geojson, locations=units, z=values.to_numpy(), zmin=0, zmax=float(values.max()) if len(values) else 1.0,
        colorscale="Reds", marker=dict(opacity=0.65, line=CLUSTER_LINE),
        hovertemplate="<b>Cluster %{location}</b><br>RMSE: %{z:.2f}<extra></extra>",
        colorbar=dict(title=dict(text="RMSE"), thickness=14, len=0.75),
    )
    fig = go.Figure(data=[choropleth, _marker_labels(units, extras, texts)])
    return _map_layout(fig, extras, height)
