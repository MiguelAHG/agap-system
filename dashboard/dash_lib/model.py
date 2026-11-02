"""
Prepares the numbers that the dashboard shows. It only uses pandas and numpy, with no Streamlit or Plotly, so it can be
tested on its own. charts.py only draws what these functions return.

Terms:
- unit: a cluster name such as "A", or CITY for the whole city
- origin week: the week a forecast is made (the current week)
- target week: the week a forecast is about (origin week plus horizon)
"""

from __future__ import annotations

import dataclasses
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
BASE_LAST_WEEK = pd.Timestamp("2026-03-20") # last week of the base disease data (12th week of 2026)
BASE_LAST_ORIGIN = pd.Timestamp("2025-12-26") # last current week that can be chosen with only the base data
TREND_CATEGORIES = ("cases", "climatic", "environmental", "population")
CATALOG_COLUMNS = ["name", "base", "base_display", "lag", "category", "unit", "static"]


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
    """Last day of the week that starts on ts."""
    return pd.Timestamp(ts) + pd.Timedelta(days=6)


def fmt_span(ts) -> str:
    """Example: Jul 10, 2026 - Jul 16, 2026"""
    return f"{fmt_week(ts)} - {fmt_week(week_end(ts))}"


def fmt_span_short(ts) -> str:
    """Example: Jul 10-Jul 16 (no year)"""
    return f"{pd.Timestamp(ts):%b %d}-{week_end(ts):%b %d}"


def iso_span(ts) -> str:
    """Example: 2026-07-10 to 2026-07-16"""
    return f"{pd.Timestamp(ts):%Y-%m-%d} to {week_end(ts):%Y-%m-%d}"


def fmt_num(value, decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def pct_change(current, previous):
    """Percent change. Returns None when it cannot be computed."""
    if previous is None or current is None or pd.isna(previous) or pd.isna(current) or previous <= 0:
        return None
    return (current - previous) / previous * 100.0


@dataclass
class DiseaseData:
    key: str
    label: str
    horizons: list
    clusters: list
    levels: list
    models: dict # horizon -> model text
    catalog: pd.DataFrame # public features, with name, base, display name, lag, category, unit and static
    forecast_long: pd.DataFrame | None # restricted, None if forecasts are not available
    fc: dict # (horizon, unit) -> table indexed by origin week
    actuals: pd.DataFrame | None # true counts by week, one column per cluster plus CITY (restricted)
    features: pd.DataFrame | None # long table with week, unit and one column per feature
    performance: pd.DataFrame | None # restricted, None if not available
    case_lags: list = field(default_factory=list) # lags in weeks of the disease count that the models use as inputs
    version: str = "full" # says which disease data this object has, used in cache keys
    _wide: dict = field(default_factory=dict) # features that are not disease counts, shared by all sessions
    _case: dict = field(default_factory=dict) # disease count series, only for this object

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
        """Table of one feature, with weeks as rows and units as columns. The result is cached."""
        if name in set(self.catalog.loc[self.catalog["category"] == "cases", "name"]):
            if name not in self._case:
                self._case[name] = case_wide(self.actuals, name)
            return self._case[name]
        if name not in self._wide:
            self._wide[name] = (self.features.pivot(index="week", columns="unit", values=name).sort_index())
        return self._wide[name]


def case_catalog(case_lags) -> pd.DataFrame:
    """
    Catalog rows for the disease counts. Includes weekly cases (this week and each lag the models use), then the 4-week total which is not a feature used in the models but is useful for the user to know
    """
    rows = [("CASES", "CASES", "Weekly cases", 0, "cases", "", False)]
    for k in sorted({int(x) for x in case_lags} - {0}):
        rows.append((f"CASES_lag_{k}", "CASES", "Weekly cases", k, "cases", "", False))
    rows.append(("CASES4", "CASES4", "Cases in the past 4 weeks", 0, "cases", "", False))
    return pd.DataFrame(rows, columns=CATALOG_COLUMNS)


def _four_week_sum(wide: pd.DataFrame) -> pd.DataFrame:
    summed = wide.rolling(4, min_periods=4).sum()
    span = pd.Series(wide.index, index=wide.index).diff(3).dt.days
    summed.loc[~(span <= 21).to_numpy()] = np.nan # a window across the 2020-21 gap is not a real 4-week total
    return summed


def case_wide(actuals: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Case counts by inference week. For a lagged series, the value at week w is the count of week w minus the lag.
    It is empty (NaN) where the data has gaps.
    """
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
    if units is not None: # unit text of each variable, from variable_units.json
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


def complete_up_to(counts: pd.DataFrame, start) -> pd.Timestamp:
    """Last week such that all weeks from start up to it are in counts."""
    week = pd.Timestamp(start)
    step = pd.Timedelta(days=7)
    while week + step in counts.index:
        week += step
    return week


def session_view(d: DiseaseData, added: pd.DataFrame | None = None, skip: bool = False, version: str = "base",
                 base_last_week=None, base_last_origin=None) -> DiseaseData:
    """
    The disease data that one session sees. It has the base data (true counts up to the last base week) plus any added
    weeks. added has cluster totals by week from an upload. With skip, the weeks after the base data in the full data are
    added instead. Current weeks are limited to what the data covers, and forecasts made after that are hidden.
    The full object d is shared and is never changed.
    """
    if not d.has_actuals or not d.fc:
        return d
    base_last_week = pd.Timestamp(base_last_week) if base_last_week is not None else BASE_LAST_WEEK
    base_last_origin = pd.Timestamp(base_last_origin) if base_last_origin is not None else BASE_LAST_ORIGIN
    full = d.actuals
    base = full.loc[:base_last_week]
    if skip:
        added = full.loc[full.index > base_last_week]
    last_origin = d.fc[(d.horizons[0], CITY)].index.max()
    if added is not None and len(added):
        added = added.reindex(columns=base.columns)
        merged = pd.concat([base[~base.index.isin(added.index)], added]).sort_index()
        limit = min(complete_up_to(merged, base_last_week), last_origin)
    else:
        merged, limit = base, base_last_origin
    fc = {k: f[f.index <= limit] for k, f in d.fc.items()}
    forecast_long = None if d.forecast_long is None else d.forecast_long[d.forecast_long["origin_week"] <= limit]
    return dataclasses.replace(d, actuals=merged, fc=fc, forecast_long=forecast_long, version=version, _case={})


def origin_weeks(d: DiseaseData) -> pd.DatetimeIndex:
    """Weeks in which a forecast can be made. All horizons share them."""
    return d.fc[(d.horizons[0], CITY)].index


def get_forecast(d: DiseaseData, horizon: int, unit: str, origin):
    frame = d.fc.get((horizon, unit))
    if frame is None or origin not in frame.index:
        return None
    return frame.loc[origin]


def key_indicators(d: DiseaseData, origin, interval_level: float = 0.8) -> dict:
    """
    Numbers for the top of the overview page. Has recent true counts (up to and including the current week) and the
    forecasts made in the current week, for every horizon. It returns plain numbers; supposed to be that the page formats them.
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
    """Barangay names of one cluster in several columns, filled top to bottom, to use less height."""
    names = sorted(barangays.loc[barangays["cluster"] == cluster, "barangay"].tolist())
    n_cols = max(2, math.ceil(len(names) / target_rows))
    n_rows = math.ceil(len(names) / n_cols) if names else 0
    padded = names + [""] * (n_rows * n_cols - len(names))
    grid = {f"Barangays ({c * n_rows + 1}-{min((c + 1) * n_rows, len(names))})": padded[c * n_rows: (c + 1) * n_rows]
            for c in range(n_cols) if c * n_rows < len(names)}
    return pd.DataFrame(grid)


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
    Label positions for values (sorted ascending). Each label is as close to its value as possible, while neighbors
    stay at least min_gap apart.
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
    One row for each number printed on the bar chart: horizon, kind (lo, pred or hi), level, value, label_y and text.
    Labels of the same horizon are pushed apart so they do not overlap. The text always shows the real value.
    """
    gap = gap_fraction * y_span
    n = len(level_keys)
    records = []
    for _, row in bars.iterrows():
        items = [("pred", "", float(row["pred"]), n)]
        for rank, k in enumerate(reversed(level_keys)): # widest level at the bottom
            items.append(("lo", k, float(row[f"lo{k}"]), rank))
        for rank, k in enumerate(level_keys): # narrowest level at the bottom
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


def timeseries_data(d: DiseaseData, horizon: int, unit: str, origin, hide_future: bool = True) -> dict:
    """
    Arrays for the line chart, with the target week on the x axis. With hide_future, true counts stop at the current week
    and forecasts stop at the week predicted from the current week.
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
    start = max(end - pd.Timedelta(days=365), full_start) # starting view: the latest 12 months
    return {
        "full_range": (full_start, full_end), "window": (start, end),
        "weeks": weeks.to_numpy(), "pred": frame["pred"].to_numpy(), "actual": actual, "bands": bands,
        "show_actual": show_actual, "show_forecast": show_forecast,
        "origin": origin, "predicted_week": predicted_week,
        "y_max": float(np.nanmax(y_candidates)) * 1.05,
    }


def lag_label(lag: int) -> str:
    return "Current week" if lag == 0 else f"{lag} week{'s' if lag != 1 else ''} before the inference week"


def variable_axis_title(d: DiseaseData, feature: str) -> str:
    """
    Variable name with its unit in parentheses, such as Maximum temperature (°C). No parentheses if there is no unit.
    """
    row = d.catalog[d.catalog["name"] == feature].iloc[0]
    return f"{row['base_display']} ({row['unit']})" if row["unit"] else row["base_display"]


def feature_title(d: DiseaseData, feature: str) -> str:
    """Chart title for one feature, such as Rainfall (mm/day), 7 weeks before the inference week."""
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
    """Turns 'rf (nonupsampled)' into 'Random Forest (RF), non-upsampled'. Unknown names are shown as they are."""
    name, _, rest = text.partition(" (")
    rest = rest.rstrip(")")
    training = {"nonupsampled": "non-upsampled", "upsampled": "upsampled"}.get(rest, rest)
    full = MODEL_NAMES.get(name.strip().lower(), name)
    return f"{full}, {training}" if training else full


def time_axis_label(d: DiseaseData, feature: str) -> str:
    """
    Returns 'Inference week' for a lagged feature, because its value is from earlier than the week shown. Otherwise 'Week'.
    """
    lag = int(d.catalog.loc[d.catalog["name"] == feature, "lag"].iloc[0])
    return "Inference week" if lag > 0 else "Week"


def base_options(d: DiseaseData, category: str, include_static: bool) -> list:
    """
    List of (base key, display name) for the variables of a category. Static variables (that do not change over time)
    are included only if asked.
    """
    c = d.catalog[d.catalog["category"] == category]
    if not include_static:
        c = c[~c["static"]]
    c = c.drop_duplicates("base")
    pairs = list(zip(c["base"], c["base_display"]))
    return pairs if category == "cases" else sorted(pairs, key=lambda t: t[1].lower())


def series_options(d: DiseaseData, base: str, include_static: bool) -> list:
    """
    List of (feature name, label) for one variable: the current week first, then the lags from smallest to largest.
    """
    c = d.catalog[d.catalog["base"] == base]
    if not include_static:
        c = c[~c["static"]]
    c = c.sort_values("lag")
    return [(n, lag_label(int(l))) for n, l in zip(c["name"], c["lag"])]


def trend_categories(d: DiseaseData) -> list:
    return [c for c in TREND_CATEGORIES if base_options(d, c, include_static=False)]


def break_gaps(series: pd.Series, max_days: int = 7) -> pd.Series:
    """Adds a NaN wherever two weeks are more than max_days apart, so a line is not drawn across a gap."""
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
    """Returns a month-by-year table of monthly means, and the mean of each month over all years."""
    clean = series.dropna()
    table = clean.groupby([clean.index.year, clean.index.month]).mean().unstack(0)
    return table, table.mean(axis=1)


@dataclass
class MapFrames:
    title: str
    colorbar_title: str
    weeks: list
    units: list
    z: np.ndarray # shape [frames, units]
    custom: np.ndarray # shape [frames, units, values]
    hover_lines: list # (label, template with {i:.1f} placeholders)
    text: list # [frame][unit], the text under each cluster marker
    zmin: float
    zmax: float
    colorscale: str
    initial_index: int
    week_noun: str = "Week" # "Inference week" if the variable is lagged

    @property
    def animated(self) -> bool:
        return len(self.weeks) > 1


def available_categories(d: DiseaseData) -> list:
    """List of (key, label) in display order. Forecast comes first, so it is the default."""
    keys = ["forecast"] if d.has_forecasts else []
    keys += [c for c in TREND_CATEGORIES if (d.catalog["category"] == c).any()]
    return [(k, CATEGORY_LABELS[k]) for k in keys]


def variable_options(d: DiseaseData, category: str) -> list:
    """
    List of (variable key, label) for a map category. For feature categories these are the variable bases.
    The series (current week or a lag) is chosen next with series_options, and map_frames receives the resulting feature name.
    """
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
        if np.nanmax(np.nanstd(z, axis=0)) == 0: # does not change over time, so one frame is enough
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


def performance_by_cluster(d: DiseaseData, horizon: int, scope: str) -> tuple:
    """Returns the RMSE of each cluster (a Series), the pooled RMSE, and the RMSE of the citywide totals."""
    p = d.performance
    p = p[(p["horizon"] == horizon) & (p["scope"] == scope)].set_index("unit")["rmse"]
    return p.reindex(d.clusters), float(p.get("POOLED", np.nan)), float(p.get(CITY, np.nan))
