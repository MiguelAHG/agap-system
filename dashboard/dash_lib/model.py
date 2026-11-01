"""
model.py

All data shaping for the dashboard. Pure pandas / numpy: no Streamlit or Plotly imports, so every function here can be
unit-tested and reused. The chart code (charts.py) only draws what these functions return.

Conventions
  unit        a cluster name ("A", "B", ...) or CITY for citywide totals
  origin week the week a forecast is made ("current week")
  target week the week a forecast is about (origin week + horizon)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CITY = "CITY"
CITY_LABEL = "Citywide"

CATEGORY_LABELS = {
    "forecast": "Forecasted disease case counts",
    "cases": "Historical disease case counts",
    "climatic": "Historical climatic variables",
    "environmental": "Historical environmental variables",
    "population": "Historical population variables",
}
TREND_CATEGORIES = ("cases", "climatic", "environmental", "population")
CATALOG_COLUMNS = ["name", "base", "base_display", "lag", "category", "unit", "static"]


# ----------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------

def level_key(level: float) -> str:
    return f"{level * 100:g}"


def level_label(level: float) -> str:
    return f"{level_key(level)}%"


def unit_label(unit: str) -> str:
    return CITY_LABEL if unit == CITY else f"Cluster {unit}"


def horizon_label(h: int) -> str:
    return f"{h} week{'s' if h != 1 else ''} ahead"


def fmt_week(ts) -> str:
    return pd.Timestamp(ts).strftime("%b %d, %Y")


def week_end(ts) -> pd.Timestamp:
    """Last day of a week that starts on ts."""
    return pd.Timestamp(ts) + pd.Timedelta(days=6)


def fmt_span(ts) -> str:
    """'Jul 10, 2026 - Jul 16, 2026'"""
    return f"{fmt_week(ts)} - {fmt_week(week_end(ts))}"


def fmt_span_short(ts) -> str:
    """'Jul 10-Jul 16' (no year)"""
    return f"{pd.Timestamp(ts):%b %d}-{week_end(ts):%b %d}"


def iso_span(ts) -> str:
    """'2026-07-10 to 2026-07-16'"""
    return f"{pd.Timestamp(ts):%Y-%m-%d} to {week_end(ts):%Y-%m-%d}"


def fmt_num(value, decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def pct_change(current, previous):
    """Percent change, or None when it is not meaningful."""
    if previous is None or current is None or pd.isna(previous) or pd.isna(current) or previous <= 0:
        return None
    return (current - previous) / previous * 100.0


# ----------------------------------------------------------------------------------------------
# Container
# ----------------------------------------------------------------------------------------------

@dataclass
class DiseaseData:
    key: str
    label: str
    horizons: list
    clusters: list
    levels: list
    models: dict                       # horizon -> text
    catalog: pd.DataFrame              # public features: name, base, base_display, lag, category, unit, static
    forecast_long: pd.DataFrame | None  # restricted; None when forecasts are not available
    fc: dict                           # (horizon, unit) -> DataFrame indexed by origin_week
    actuals: pd.DataFrame | None       # index week, columns clusters + CITY (true counts; restricted)
    features: pd.DataFrame | None      # long: week, unit, <feature columns>
    performance: pd.DataFrame | None   # restricted; None when not available
    case_lags: list = field(default_factory=list)   # lags (weeks) of the disease count used as model inputs
    _wide: dict = field(default_factory=dict)

    @property
    def has_actuals(self) -> bool:
        return self.actuals is not None

    @property
    def has_forecasts(self) -> bool:
        return bool(self.fc)

    @property
    def units(self) -> list:
        return [CITY] + list(self.clusters)

    def feature_wide(self, name: str) -> pd.DataFrame:
        """week x unit table of one feature (cached)."""
        if name not in self._wide:
            if name in set(self.catalog.loc[self.catalog["category"] == "cases", "name"]):
                self._wide[name] = case_wide(self.actuals, name)
            else:
                self._wide[name] = (self.features.pivot(index="week", columns="unit", values=name).sort_index())
        return self._wide[name]


def case_catalog(case_lags) -> pd.DataFrame:
    """Catalog rows for the disease count series: weekly cases (current week and each modelled lag), then 4-week totals."""
    rows = [("CASES", "CASES", "Weekly cases", 0, "cases", "", False)]
    for k in sorted({int(x) for x in case_lags} - {0}):
        rows.append((f"CASES_lag_{k}", "CASES", "Weekly cases", k, "cases", "", False))
    rows.append(("CASES4", "CASES4", "Cases in the past 4 weeks", 0, "cases", "", False))
    return pd.DataFrame(rows, columns=CATALOG_COLUMNS)


def _four_week_sum(wide: pd.DataFrame) -> pd.DataFrame:
    summed = wide.rolling(4, min_periods=4).sum()
    span = pd.Series(wide.index, index=wide.index).diff(3).dt.days
    summed.loc[~(span <= 21).to_numpy()] = np.nan          # windows that cross the 2020-21 gap are not real 4-week totals
    return summed


def case_wide(actuals: pd.DataFrame, name: str) -> pd.DataFrame:
    """Case series by inference week. A lagged series at week w holds the count of week w - lag (NaN across data gaps)."""
    if name == "CASES4":
        return _four_week_sum(actuals)
    lag = int(name.rsplit("_lag_", 1)[1]) if "_lag_" in name else 0
    if lag == 0:
        return actuals
    shifted = actuals.reindex(actuals.index - pd.Timedelta(days=7 * lag))
    shifted.index = actuals.index
    return shifted


def build_disease_data(key, meta, forecasts, features, actuals, performance, units=None, case_lags=None) -> DiseaseData:
    fc = {}
    if forecasts is not None:
        for (h, unit), group in forecasts.groupby(["horizon", "unit"]):
            fc[(int(h), str(unit))] = group.drop(columns=["unit", "horizon"]).set_index("origin_week").sort_index()
    catalog = pd.DataFrame(meta["features"], columns=CATALOG_COLUMNS)
    catalog = catalog[catalog["category"] != "cases"]
    if units is not None:                                   # unit text per variable base, from variable_units.json
        catalog["unit"] = catalog["base"].map(units).fillna("")
    if case_lags is None:
        case_lags = meta.get("case_lags", [])
    if actuals is not None:
        actuals = actuals.set_index("week").sort_index()
        catalog = pd.concat([case_catalog(case_lags), catalog], ignore_index=True)
    return DiseaseData(
        key=key, label=meta["label"], horizons=list(meta["horizons"]), clusters=list(meta["clusters"]),
        levels=list(meta["levels"]), models={int(h): t for h, t in meta["models_by_horizon"].items()},
        catalog=catalog, forecast_long=forecasts, fc=fc, actuals=actuals, features=features, performance=performance,
        case_lags=list(case_lags),
    )


# ----------------------------------------------------------------------------------------------
# Weeks
# ----------------------------------------------------------------------------------------------

def origin_weeks(d: DiseaseData) -> pd.DatetimeIndex:
    """Weeks a forecast can be made (all horizons share them)."""
    return d.fc[(d.horizons[0], CITY)].index


def get_forecast(d: DiseaseData, horizon: int, unit: str, origin):
    frame = d.fc.get((horizon, unit))
    if frame is None or origin not in frame.index:
        return None
    return frame.loc[origin]


# ----------------------------------------------------------------------------------------------
# Key indicators
# ----------------------------------------------------------------------------------------------

def key_indicators(d: DiseaseData, origin, interval_level: float = 0.8) -> dict:
    """
    Recent past (true counts, up to and including the current week) and the forecasts made in the current week, for
    every horizon. Returns plain numbers; formatting happens in the view.
    """
    out = {"past": None, "future": []}

    if d.has_actuals and origin in d.actuals.index:
        city = d.actuals[CITY]
        pos = city.index.get_loc(origin)
        current = float(city.iloc[pos])
        previous = float(city.iloc[pos - 1]) if pos >= 1 else None
        last4 = float(city.iloc[pos - 3: pos + 1].sum()) if pos >= 3 else None
        prev4 = float(city.iloc[pos - 7: pos - 3].sum()) if pos >= 7 else None
        top = None
        if pos >= 3:
            by_cluster = d.actuals[d.clusters].iloc[pos - 3: pos + 1].sum()
            top = (str(by_cluster.idxmax()), float(by_cluster.max()))
        out["past"] = {
            "week_cases": current, "week_change": pct_change(current, previous),
            "last4": last4, "last4_change": pct_change(last4, prev4), "top_cluster": top,
        }

    key = level_key(interval_level)
    for h in d.horizons:
        row = get_forecast(d, h, CITY, origin)
        if row is None:
            continue
        out["future"].append({
            "horizon": h, "target_week": row["target_week"], "pred": float(row["pred"]),
            "lo": float(row[f"lo{key}"]), "hi": float(row[f"hi{key}"]), "level": interval_level,
        })

    h0 = d.horizons[0]
    best = None
    for c in d.clusters:
        r = get_forecast(d, h0, c, origin)
        if r is not None and (best is None or r["pred"] > best[1]["pred"]):
            best = (c, r)
    out["top_forecast"] = None if best is None else {
        "horizon": h0, "cluster": best[0], "pred": float(best[1]["pred"]),
        "lo": float(best[1][f"lo{key}"]), "hi": float(best[1][f"hi{key}"]), "level": interval_level,
    }
    return out


# ----------------------------------------------------------------------------------------------
# Per-cluster forecast table (clusters as columns, so it stays short)
# ----------------------------------------------------------------------------------------------

def cluster_forecast_table(d: DiseaseData, origin, horizon: int) -> pd.DataFrame:
    rows = {"Point Forecast": {}}
    for level in d.levels:
        rows[f"{level_label(level)} interval"] = {}
    for unit in d.units:
        r = get_forecast(d, horizon, unit, origin)
        label = unit_label(unit)
        if r is None:
            for name in rows:
                rows[name][label] = "N/A"
            continue
        rows["Point Forecast"][label] = f"{r['pred']:.1f}"
        for level in d.levels:
            k = level_key(level)
            rows[f"{level_label(level)} interval"][label] = f"{r[f'lo{k}']:.1f} to {r[f'hi{k}']:.1f}"
    table = pd.DataFrame(rows).T
    table.index.name = None
    return table


def barangay_grid(barangays: pd.DataFrame, cluster: str, target_rows: int = 6) -> pd.DataFrame:
    """Barangay names of a cluster laid out in several columns (filled top to bottom) to save vertical space."""
    names = sorted(barangays.loc[barangays["cluster"] == cluster, "barangay"].tolist())
    n_cols = max(2, math.ceil(len(names) / target_rows))
    n_rows = math.ceil(len(names) / n_cols) if names else 0
    padded = names + [""] * (n_rows * n_cols - len(names))
    grid = {f"Barangays ({c * n_rows + 1}-{min((c + 1) * n_rows, len(names))})": padded[c * n_rows: (c + 1) * n_rows]
            for c in range(n_cols) if c * n_rows < len(names)}
    return pd.DataFrame(grid)


# ----------------------------------------------------------------------------------------------
# Bar chart of all horizons
# ----------------------------------------------------------------------------------------------

def horizon_bars(d: DiseaseData, origin, unit: str) -> pd.DataFrame:
    rows = []
    for h in d.horizons:
        r = get_forecast(d, h, unit, origin)
        if r is None:
            continue
        row = {"horizon": h, "target_week": r["target_week"], "pred": float(r["pred"])}
        for level in d.levels:
            k = level_key(level)
            row[f"lo{k}"], row[f"hi{k}"] = float(r[f"lo{k}"]), float(r[f"hi{k}"])
        rows.append(row)
    return pd.DataFrame(rows)


def spread_positions(values: list, min_gap: float) -> list:
    """
    Label positions for `values` (ascending): as close to the values as possible while neighbours are at least
    min_gap apart (isotonic regression on the gap-shifted values).
    """
    shifted = [v - i * min_gap for i, v in enumerate(values)]
    blocks = []
    for s in shifted:
        blocks.append([s, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s2, c2 = blocks.pop()
            blocks[-1][0] += s2
            blocks[-1][1] += c2
    fitted = []
    for total, count in blocks:
        fitted += [total / count] * count
    return [f + i * min_gap for i, f in enumerate(fitted)]


def bar_label_layout(bars: pd.DataFrame, level_keys: list, y_span: float, gap_fraction: float = 0.05) -> pd.DataFrame:
    """
    One row per number printed in the bar chart: horizon, kind (lo / pred / hi), level key, value, label_y, text.
    Labels of one horizon are pushed apart vertically so numbers never overlap; text always shows the true value.
    """
    gap = gap_fraction * y_span
    n = len(level_keys)
    records = []
    for _, row in bars.iterrows():
        items = [("pred", "", float(row["pred"]), n)]
        for rank, k in enumerate(reversed(level_keys)):              # widest level lowest
            items.append(("lo", k, float(row[f"lo{k}"]), rank))
        for rank, k in enumerate(level_keys):                        # narrowest level lowest
            items.append(("hi", k, float(row[f"hi{k}"]), n + 1 + rank))
        items.sort(key=lambda t: (t[2], t[3]))
        positions = spread_positions([t[2] for t in items], gap)
        for (kind, k, value, _), y in zip(items, positions):
            text = f"{value:.1f}" if kind == "pred" else f"{k}%: {value:.1f}"
            records.append({"horizon": int(row["horizon"]), "kind": kind, "level": k, "value": value,
                            "label_y": y, "text": text})
    return pd.DataFrame(records)


def nice_ticks(y_max: float, target: int = 6) -> list:
    if y_max <= 0:
        return [0.0, 1.0]
    raw = y_max / target
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    count = int(math.ceil(y_max / step)) + 1
    return [round(i * step, 10) for i in range(count)]


# ----------------------------------------------------------------------------------------------
# Time series (actual vs forecast)
# ----------------------------------------------------------------------------------------------

def timeseries_data(d: DiseaseData, horizon: int, unit: str, origin, hide_future: bool = True) -> dict:
    """
    Arrays for the forecast line chart, x = target week. With hide_future, true counts stop at the current week and
    forecasts stop at the week being predicted from the current week.
    """
    frame = d.fc[(horizon, unit)].reset_index()
    origin = pd.Timestamp(origin)
    target_of_origin = frame.loc[frame["origin_week"] == origin, "target_week"]
    predicted_week = target_of_origin.iloc[0] if len(target_of_origin) else origin + pd.Timedelta(weeks=horizon)

    weeks = frame["target_week"]
    actual = None
    if d.has_actuals:
        actual = d.actuals[unit].reindex(weeks.values).to_numpy()

    show_actual = (weeks <= origin).to_numpy() if hide_future else np.ones(len(weeks), bool)
    show_forecast = (weeks <= predicted_week).to_numpy() if hide_future else np.ones(len(weeks), bool)

    bands = []
    for level in d.levels:
        k = level_key(level)
        bands.append({"level": level, "label": level_label(level),
                      "lo": frame[f"lo{k}"].to_numpy(), "hi": frame[f"hi{k}"].to_numpy()})

    y_candidates = [frame[f"hi{level_key(d.levels[-1])}"].max()]
    if actual is not None:
        y_candidates.append(np.nanmax(actual))
    full_start, full_end = pd.Timestamp(weeks.min()) - pd.Timedelta(days=14), pd.Timestamp(weeks.max()) + pd.Timedelta(days=60)
    end = min(predicted_week + pd.Timedelta(days=58), full_end) if hide_future else full_end
    start = max(end - pd.Timedelta(days=365), full_start)                  # default view: the latest year-long window
    return {
        "full_range": (full_start, full_end), "window": (start, end),
        "weeks": weeks.to_numpy(), "pred": frame["pred"].to_numpy(), "actual": actual, "bands": bands,
        "show_actual": show_actual, "show_forecast": show_forecast,
        "origin": origin, "predicted_week": predicted_week,
        "y_max": float(np.nanmax(y_candidates)) * 1.05,
    }


# ----------------------------------------------------------------------------------------------
# Historical trends
# ----------------------------------------------------------------------------------------------

def lag_label(lag: int) -> str:
    return "Current week" if lag == 0 else f"{lag} week{'s' if lag != 1 else ''} before the inference week"


def variable_axis_title(d: DiseaseData, feature: str) -> str:
    """Variable name with its unit in parentheses, e.g. 'Maximum temperature (°C)'. No parentheses when unitless."""
    row = d.catalog[d.catalog["name"] == feature].iloc[0]
    return f"{row['base_display']} ({row['unit']})" if row["unit"] else row["base_display"]


def feature_title(d: DiseaseData, feature: str) -> str:
    """Chart title for one feature column, e.g. 'Rainfall (mm/day), 7 weeks before the inference week'."""
    row = d.catalog[d.catalog["name"] == feature].iloc[0]
    name = variable_axis_title(d, feature)
    return name if row["lag"] == 0 else f"{name}, {lag_label(int(row['lag']))}"


def feature_lag(d: DiseaseData, feature: str) -> int:
    return int(d.catalog.loc[d.catalog["name"] == feature, "lag"].iloc[0])


MODEL_NAMES = {
    "rf": "Random Forest (RF)",
    "xgboost": "eXtreme Gradient Boosting (XGBoost)",
    "svm": "Support Vector Machine (SVM)",
    "lstm": "Long Short-Term Memory (LSTM)",
    "stgcn": "Spatio-Temporal Graph Convolutional Network (STGCN)",
}


def model_text(text: str) -> str:
    """'rf (nonupsampled)' -> 'Random Forest (RF), non-upsampled'. Unknown names are shown as given."""
    name, _, rest = text.partition(" (")
    rest = rest.rstrip(")")
    training = {"nonupsampled": "non-upsampled", "upsampled": "upsampled"}.get(rest, rest)
    full = MODEL_NAMES.get(name.strip().lower(), name)
    return f"{full}, {training}" if training else full


def time_axis_label(d: DiseaseData, feature: str) -> str:
    """'Inference week' when the feature is lagged (values are from earlier than the week shown), else 'Week'."""
    lag = int(d.catalog.loc[d.catalog["name"] == feature, "lag"].iloc[0])
    return "Inference week" if lag > 0 else "Week"


def base_options(d: DiseaseData, category: str, include_static: bool) -> list:
    """[(base key, display name)] of the variables in a category. Static ones (constant over time) are optional."""
    c = d.catalog[d.catalog["category"] == category]
    if not include_static:
        c = c[~c["static"]]
    c = c.drop_duplicates("base")
    pairs = list(zip(c["base"], c["base_display"]))
    return pairs if category == "cases" else sorted(pairs, key=lambda t: t[1].lower())


def series_options(d: DiseaseData, base: str, include_static: bool) -> list:
    """[(feature name, label)] of the series of one variable: current week first, then lags in increasing order."""
    c = d.catalog[d.catalog["base"] == base]
    if not include_static:
        c = c[~c["static"]]
    c = c.sort_values("lag")
    return [(n, lag_label(int(l))) for n, l in zip(c["name"], c["lag"])]


def trend_categories(d: DiseaseData) -> list:
    return [c for c in TREND_CATEGORIES if base_options(d, c, include_static=False)]


def break_gaps(series: pd.Series, max_days: int = 7) -> pd.Series:
    """Insert a NaN wherever consecutive weeks are further apart than max_days, so lines do not join across gaps."""
    index = series.index
    if len(index) < 2:
        return series
    days = np.diff(index.values).astype("timedelta64[D]").astype(int)
    gaps = np.flatnonzero(days > max_days)
    if len(gaps) == 0:
        return series
    breaks = pd.Series(np.nan, index=index[gaps] + pd.Timedelta(days=1))
    return pd.concat([series, breaks]).sort_index()


def trend_series(d: DiseaseData, feature: str, unit: str) -> pd.Series:
    return break_gaps(d.feature_wide(feature)[unit].dropna())


def seasonal_profile(series: pd.Series) -> tuple:
    """(month x year table of monthly means, mean over years by month)."""
    clean = series.dropna()
    table = clean.groupby([clean.index.year, clean.index.month]).mean().unstack(0)
    return table, table.mean(axis=1)


# ----------------------------------------------------------------------------------------------
# Map frames
# ----------------------------------------------------------------------------------------------

@dataclass
class MapFrames:
    title: str
    colorbar_title: str
    weeks: list
    units: list
    z: np.ndarray                  # [F, U]
    custom: np.ndarray             # [F, U, K]
    hover_lines: list              # [(label, template with {i:.1f} placeholders)]
    text: list                     # [F][U] label under each cluster marker
    zmin: float
    zmax: float
    colorscale: str
    initial_index: int
    week_noun: str = "Week"                 # "Inference week" when the variable is lagged

    @property
    def animated(self) -> bool:
        return len(self.weeks) > 1


def available_categories(d: DiseaseData) -> list:
    """[(key, label)] in display order (forecast first, so it is the default)."""
    keys = ["forecast"] if d.has_forecasts else []
    keys += [c for c in TREND_CATEGORIES if (d.catalog["category"] == c).any()]
    return [(k, CATEGORY_LABELS[k]) for k in keys]


def variable_options(d: DiseaseData, category: str) -> list:
    """[(variable key, human readable label)] for a map category. Feature categories list variable bases; the series
    (current week or a lag) is chosen next with series_options, and map_frames takes the resulting feature name."""
    if category == "forecast":
        return [(str(h), horizon_label(h)) for h in d.horizons]
    return base_options(d, category, include_static=True)


def decimals_for(values: np.ndarray) -> int:
    finite = values[np.isfinite(values)]
    if finite.size == 0 or np.allclose(finite, np.round(finite)):
        return 0
    biggest = np.abs(finite).max()
    return 1 if biggest >= 100 else (2 if biggest >= 1 else 3)


def _text_labels(z: np.ndarray, units: list, decimals: int) -> list:
    return [[f"{u}<br>{'' if np.isnan(v) else f'{v:,.{decimals}f}'}" for u, v in zip(units, row)] for row in z]


def map_frames(d: DiseaseData, category: str, variable: str) -> MapFrames:
    units = list(d.clusters)
    is_feature = category in TREND_CATEGORIES
    options = {variable: feature_title(d, variable)} if is_feature else dict(variable_options(d, category))
    title = f"{CATEGORY_LABELS[category]}: {options[variable]}"
    week_noun = "Inference week" if is_feature and feature_lag(d, variable) > 0 else "Week"

    if category == "forecast":
        h = int(variable)
        long = d.forecast_long
        long = long[(long["horizon"] == h) & (long["unit"] != CITY)]
        columns = ["pred"] + [f"{p}{level_key(l)}" for l in d.levels for p in ("lo", "hi")]
        wide = {c: long.pivot(index="target_week", columns="unit", values=c).sort_index()[units] for c in columns}
        weeks = list(wide["pred"].index)
        z = wide["pred"].to_numpy(dtype="float64")
        parts = [wide[c].to_numpy(dtype="float64") for c in columns]
        hover = [("Forecast", "{0:.1f}")]
        for i, level in enumerate(d.levels):
            hover.append((f"{level_label(level)} interval", f"{{{1 + 2 * i}:.1f}} to {{{2 + 2 * i}:.1f}}"))
        if d.has_actuals:
            actual = d.actuals[units].reindex(weeks).to_numpy(dtype="float64")
            parts.append(actual)
            hover.append(("Actual cases", f"{{{len(parts) - 1}:.0f}}"))
        custom = np.stack(parts, axis=-1)
        decimals, scale, unit_text = 1, "YlOrRd", "Cases"
    else:
        wide = d.feature_wide(variable)[units].dropna(how="all")
        weeks, z = list(wide.index), wide.to_numpy(dtype="float64")
        if np.nanmax(np.nanstd(z, axis=0)) == 0:                       # constant over time: one frame
            weeks, z = weeks[-1:], z[-1:]
        decimals = decimals_for(z)
        custom, hover = z[..., None], [(options[variable], f"{{0:.{decimals}f}}")]
        row = d.catalog[d.catalog["name"] == variable].iloc[0]
        scale, unit_text = ("YlOrRd", "Cases") if category == "cases" else ("Viridis", row["unit"] or "")

    finite = z[np.isfinite(z)]
    zmin = 0.0 if category in ("forecast", "cases") else float(finite.min())
    zmax = float(finite.max())
    if zmax <= zmin:
        zmax = zmin + 1.0
    return MapFrames(
        title=title, colorbar_title=unit_text, weeks=weeks, units=units,
        z=np.round(z, 4), custom=np.round(custom, 4), hover_lines=hover,
        text=_text_labels(z, units, decimals), zmin=zmin, zmax=zmax, colorscale=scale,
        initial_index=len(weeks) - 1, week_noun=week_noun,
    )


# ----------------------------------------------------------------------------------------------
# Model performance
# ----------------------------------------------------------------------------------------------

def performance_by_cluster(d: DiseaseData, horizon: int, scope: str) -> tuple:
    """(Series of RMSE by cluster, pooled RMSE, citywide-total RMSE)."""
    p = d.performance
    p = p[(p["horizon"] == horizon) & (p["scope"] == scope)].set_index("unit")["rmse"]
    return p.reindex(d.clusters), float(p.get("POOLED", np.nan)), float(p.get(CITY, np.nan))
