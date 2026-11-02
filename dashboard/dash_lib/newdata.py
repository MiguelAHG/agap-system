"""
Checks an uploaded weekly case file with the columns week_start, barangay and cases.

A week is complete when every barangay has a value for it. Missing barangay rows and missing weeks inside the uploaded
range are filled with zero and reported.

Any invalid value makes the whole file get rejected.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

GRID_ORIGIN = pd.Timestamp("2010-01-01") # weeks start on a 7-day grid that begins on this date
COLUMNS = ("week_start", "barangay", "cases")


@dataclass
class UploadResult:
    ok: bool
    errors: list = field(default_factory=list) # lines of the log shown when the file is rejected
    headline: str = "Error: There are invalid data values"
    missing_weeks: list = field(default_factory=list) # week starts (YYYY-MM-DD) where missing barangay rows were filled with zero
    counts: pd.DataFrame | None = None # cluster totals of the uploaded weeks, with CITY as an extra column
    version: str = ""
    first_week: pd.Timestamp | None = None
    last_week: pd.Timestamp | None = None


def _blank(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == ""


def _parse_count(value):
    """Returns (count, None) if the value is valid, otherwise (None, error message)."""
    if _blank(value):
        return None, "Disease case count is missing."
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return None, f"Disease case count {text} is not an integer."
    if not np.isfinite(number) or number != round(number):
        return None, f"Disease case count {text} is not an integer."
    if number < 0:
        return None, f"Disease case count {text} is negative."
    return int(number), None


def validate(raw: pd.DataFrame, barangays: pd.DataFrame, clusters: list, city: str = "CITY",
             base_last_week=None) -> UploadResult:
    """
    barangays has the columns barangay and cluster, for every barangay of the city. The checks on each row come first.
    Completeness is checked for the weeks after base_last_week (all weeks of the file if None) up to the last uploaded
    week. Rows for earlier weeks are not used.
    """
    lookup = {str(c).strip().lower(): c for c in raw.columns}
    if not set(COLUMNS) <= set(lookup):
        return UploadResult(False, errors=[f"The file must have the columns: {', '.join(COLUMNS)}."])
    df = raw[[lookup[c] for c in COLUMNS]].copy()
    df.columns = list(COLUMNS)
    if df.empty:
        return UploadResult(False, errors=["The file has no rows."])

    valid_names = set(barangays["barangay"])
    n = len(df)
    parts = [[] for _ in range(n)]
    dates = [pd.NaT] * n
    names = [None] * n
    counts = [None] * n
    for pos, (week, name, cases) in enumerate(df.itertuples(index=False)):
        if _blank(name):
            parts[pos].append("Barangay name is missing.")
        elif str(name).strip() not in valid_names:
            parts[pos].append(f"'{str(name).strip()}' is not a valid barangay name.")
        else:
            names[pos] = str(name).strip()
        if _blank(week):
            parts[pos].append("Week start is missing.")
        else:
            text = str(week).strip()
            stamp = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
            if pd.isna(stamp) or (stamp - GRID_ORIGIN).days % 7 != 0:
                parts[pos].append(f"'{text}' is not a valid week start date.")
            else:
                dates[pos] = stamp
        value, problem = _parse_count(cases)
        if problem:
            parts[pos].append(problem)
        counts[pos] = value

    seen = set()
    for pos in range(n): # the same barangay and week twice is not allowed
        if names[pos] is not None and not pd.isna(dates[pos]):
            key = (names[pos], dates[pos])
            if key in seen:
                parts[pos].append("Duplicate entry for this barangay and week.")
            seen.add(key)

    errors = [f"{pos}-th row (index {pos}): " + " ".join(p) for pos, p in enumerate(parts) if p]
    if errors:
        return UploadResult(False, errors=errors)

    frame = pd.DataFrame({"week": pd.to_datetime(dates), "barangay": names, "cases": counts})
    last = frame["week"].max()
    if base_last_week is not None:
        start = pd.Timestamp(base_last_week) + pd.Timedelta(days=7)
        if last < start:
            return UploadResult(False, headline=f"Error: The file has no weeks after {pd.Timestamp(base_last_week):%Y-%m-%d}",
                                errors=[f"The last week in the file is {last:%Y-%m-%d}."])
        frame = frame[frame["week"] >= start]
    else:
        start = frame["week"].min()
    weeks = pd.date_range(start, last, freq="7D")
    all_names = sorted(valid_names)
    wide = frame.pivot_table(index="week", columns="barangay", values="cases", aggfunc="sum")
    present = frame.pivot_table(index="week", columns="barangay", values="cases", aggfunc="count")
    wide = wide.reindex(index=weeks, columns=all_names)
    present = present.reindex(index=weeks, columns=all_names).fillna(0) > 0
    incomplete = [w for w, ok in zip(weeks, present.all(axis=1)) if not ok]
    missing_weeks = [f"{w:%Y-%m-%d}" for w in incomplete]
    wide = wide.fillna(0.0)

    cluster_of = barangays.drop_duplicates("barangay").set_index("barangay")["cluster"].reindex(all_names)
    by_cluster = wide.T.groupby(cluster_of.to_numpy()).sum().T.reindex(columns=list(clusters), fill_value=0.0)
    by_cluster[city] = by_cluster.sum(axis=1)
    by_cluster = by_cluster.astype("float64")
    by_cluster.index.name = "week"
    version = hashlib.md5(by_cluster.to_numpy().tobytes() + by_cluster.index.to_numpy().tobytes()).hexdigest()[:12]
    return UploadResult(True, missing_weeks=missing_weeks, counts=by_cluster, version=version,
                        first_week=weeks[0], last_week=weeks[-1])


def validate_csv(file, barangays: pd.DataFrame, clusters: list, city: str = "CITY", base_last_week=None) -> UploadResult:
    try:
        file.seek(0)
        raw = pd.read_csv(file, dtype=str)
    except Exception:
        return UploadResult(False, errors=["The file could not be read as a CSV."])
    return validate(raw, barangays, clusters, city, base_last_week)
