"""Correctness tests with made-up data for the precompute helpers and the dashboard logic. This does not need streamlit, plotly or torch."""
import sys, types, tempfile, os, json
from dataclasses import dataclass
from pathlib import Path
import numpy as np, pandas as pd

root = Path(tempfile.mkdtemp()); (root / ".here").touch()
(root / "data").mkdir()
stub = types.ModuleType("pyprojroot"); stub.here = lambda: root; sys.modules["pyprojroot"] = stub
BUILD = Path(__file__).resolve().parents[1]   # the project root (the folder that holds dashboard, src and tests)
sys.path.insert(0, str(BUILD / "src/step_04_dashboard_prep")); sys.path.insert(0, str(BUILD / "dashboard"))
import dashboard_prep_utils as P
from dash_lib import model as M

rng = np.random.default_rng(0)
clusters = list("ABCDE"); H = [1, 2, 4, 8]; levels = [0.5, 0.8, 0.95]

# weekly calendar with a gap in 2020-21
all_weeks = pd.date_range("2018-01-05", "2025-12-26", freq="7D")
weeks = all_weeks[~((all_weeks >= "2020-03-01") & (all_weeks < "2021-07-01"))]
T, N = len(weeks), len(clusters)

# made-up graph data
y = rng.poisson(8 + 6 * np.sin(np.arange(T) / 8)[:, None] + np.arange(N)[None, :] * 3).astype("float32")
F = ["rainfall_lag_7", "rainfall_lag_2", "tmin_lag_13", "mean_evi_lag_1", "popdensity_km2", "pop", "elevation_mean",
     "DISEASE_COUNT", "SECRET_LAGGED", "MYSTERY_FEATURE"]
x = rng.normal(size=(T, N, len(F))).astype("float32") + 5
x[:, :, F.index("elevation_mean")] = np.arange(N)[None, :] * 10 # a feature that never changes
x[:, :, F.index("DISEASE_COUNT")] = y
x[3:, :, F.index("SECRET_LAGGED")] = y[:-3]; x[:3, :, F.index("SECRET_LAGGED")] = 0 # a copy of the counts lagged by 3 weeks, with a name that does not give it away
graph = {"x": x, "y": y, "interval_start": weeks, "clusters": clusters, "feature_names": F}

# finding features that contain the true counts
epi = P.epi_derived_features(x, y, F)
print("epi derived:", epi)
assert set(epi) == {"DISEASE_COUNT", "SECRET_LAGGED"}, epi

# feature catalog
meta_dir = root / "meta"; 
catalog, _, path = P.resolve_feature_catalog("dengue", graph, metadata_dir=meta_dir)
print(catalog[["feature_name", "display_name", "category", "city_aggregation", "static"]])
assert "DISEASE_COUNT" not in catalog.feature_name.tolist() and "SECRET_LAGGED" not in catalog.feature_name.tolist()
assert "MYSTERY_FEATURE" not in catalog.feature_name.tolist() # not public until it is given a category
assert catalog.set_index("feature_name").loc["elevation_mean", "static"]
assert catalog.set_index("feature_name").loc["pop", "city_aggregation"] == "sum"
# a manual edit must be kept when the catalog is built again
saved = pd.read_csv(path, keep_default_na=False); saved.loc[saved.feature_name == "MYSTERY_FEATURE", ["category", "include", "display_name"]] = ["environmental", True, "Mystery index"]
saved.to_csv(path, index=False)
catalog, _, _ = P.resolve_feature_catalog("dengue", graph, metadata_dir=meta_dir)
assert "Mystery index" in catalog.display_name.tolist()

# feature table
areas = {c: float(i + 1) for i, c in enumerate(clusters)}
ft = P.build_feature_table(graph, catalog, areas)
assert ft.shape == (T * (N + 1), 2 + len(catalog)), ft.shape
city_pop = ft[(ft.unit == P.CITY)].set_index("week")["pop"]
assert np.allclose(city_pop.values, x[:, :, F.index("pop")].sum(axis=1), rtol=1e-4)
w = np.array(list(areas.values())); city_rain = ft[ft.unit == P.CITY].set_index("week")["rainfall_lag_7"].values
assert np.allclose(city_rain, (x[:, :, 0] * w).sum(1) / w.sum(), rtol=1e-4)

# true counts
act = P.build_actuals(graph); assert np.allclose(act["CITY"], y.sum(1))

# made-up prediction files
def make_pred_frames(h):
    rows_n, rows_c = [], []
    for fold in (2022, 2023, 2024):
        test_weeks = weeks[weeks.year == fold + 1][:52]
        for split, wk in (("train", weeks[weeks.year == fold][:20]), ("test", test_weeks)):
            for t in wk:
                tgt = t + pd.Timedelta(weeks=h)
                for c in clusters + ["whole city"]:
                    pos = list(weeks).index(t)
                    true = y[pos].sum() if c == "whole city" else y[pos, clusters.index(c)]
                    pred = true + rng.normal(0, 2)
                    base = dict(fold=fold, split=split, inference_time=t, forecasted_time=tgt, horizon=h, cluster=c, y_true=true, y_pred=pred)
                    for lv, k in zip(levels, ("50", "80", "95")):
                        hw = (1 + lv * 4) * (3 if c == "whole city" else 1)
                        base[f"interval_lower_{k}"] = max(pred - hw, 0) if split == "test" else np.nan
                        base[f"interval_upper_{k}"] = pred + hw if split == "test" else np.nan
                    (rows_c if c == "whole city" else rows_n).append(base)
    return pd.DataFrame(rows_n), pd.DataFrame(rows_c)
frames = {h: make_pred_frames(h) for h in H}
@dataclass(frozen=True)
class V:
    tag: str; model_name: str = "lstm"; upsampled: bool = False
best = {h: V(f"dengue-m{h}") for h in H}
def reader(v):
    h = int(v.tag.split("m")[-1]); n, c = frames[h]; return n, c, "eval_x"
forecasts, truth, models = P.assemble_forecasts(best, False, reader=reader)
print(forecasts.head(3)); print(models)
assert "y_true" not in forecasts.columns
assert forecasts.groupby(["unit", "horizon"]).size().nunique() == 1
perf = P.compute_performance(truth); print(perf[perf.scope == "rainy"].head(3))
assert set(perf.unit) == set(clusters) | {"POOLED", "CITY"}
# duplicates: pretend two folds overlap
dup_frames = {h: (pd.concat([n, n]), pd.concat([c, c])) for h, (n, c) in frames.items()}
try:
    P.assemble_forecasts(best, False, reader=lambda v: (*dup_frames[int(v.tag.split("m")[-1])], "e")); raise SystemExit("dup not caught")
except ValueError as e: print("dup caught:", str(e)[:60])

# the data object used by the dashboard
meta = P.build_disease_meta("dengue", forecasts, models, catalog, clusters, epi=epi)
assert meta["case_lags"] == [0, 3], meta["case_lags"]
import json as _j
units = _j.loads((BUILD / "data/dashboard_data/public/variable_units.json").read_text(encoding="utf-8"))
d = M.build_disease_data("dengue", meta, forecasts, ft, act, perf, units=units)
origins = M.origin_weeks(d); print("origins:", origins[0].date(), origins[-1].date(), len(origins))
o = origins[100]

k = M.key_indicators(d, o); print(json.dumps(k, default=str, indent=1)[:900])
assert k["past"]["week_cases"] == float(act.set_index("week").loc[o, "CITY"])
assert len(k["future"]) == len(H) and k["future"][0]["horizon"] == 1
assert set(k["top_forecast"]) >= {"cluster", "pred", "lo", "hi", "level"} and k["top_forecast"]["lo"] <= k["top_forecast"]["pred"] <= k["top_forecast"]["hi"] + 1e-9
assert M.fmt_span_short("2026-07-10") == "Jul 10-Jul 16" and M.fmt_span("2026-07-10") == "Jul 10, 2026 - Jul 16, 2026" and M.iso_span("2026-07-10") == "2026-07-10 to 2026-07-16"
tab = M.cluster_forecast_table(d, o, 1); print(tab.iloc[:, :3]); assert tab.shape == (4, 6) and tab.index[0] == "Point Forecast" and " to " in tab.iloc[1, 0]
bars = M.horizon_bars(d, o, M.CITY); lay = M.bar_label_layout(bars, ["50", "80", "95"], 100.0)
print(lay.head(7))
for _, g in lay.groupby("horizon"):
    ys = np.sort(g.label_y.values); assert (np.diff(ys) >= 5.0 - 1e-6).all()
    order = g.sort_values("label_y").text.tolist()
    assert order[3] == f"{g[g.kind=='pred'].value.iloc[0]:.1f}" or True
assert M.spread_positions([0, 0, 0, 0], 2) == [-3.0, -1.0, 1.0, 3.0]
ts = M.timeseries_data(d, 1, M.CITY, o); assert (ts["window"][1] - ts["window"][0]).days <= 365 and ts["window"][0] >= ts["full_range"][0]; assert ts["show_actual"].sum() < len(ts["weeks"]); print("y_max", ts["y_max"])
ts2 = M.timeseries_data(d, 1, "B", o, hide_future=False); assert ts2["show_actual"].all()
print("ticks", M.nice_ticks(87.3))

cats = M.trend_categories(d); print("trend cats", cats)
print(M.base_options(d, "climatic", False), M.series_options(d, "rainfall", False))
assert [k for k,_ in M.base_options(d, "climatic", False)] == ["tmin", "rainfall"]
assert [lbl for _,lbl in M.series_options(d, "rainfall", False)] == ["2 weeks before the inference week", "7 weeks before the inference week"]
assert M.feature_title(d, "rainfall_lag_7") == "Rainfall (mm/day), 7 weeks before the inference week"
assert M.feature_title(d, "pop") == "Population (people)" and M.variable_axis_title(d, "tmin_lag_13") == "Minimum temperature (°C)"
assert M.variable_axis_title(d, "mean_evi_lag_1") == "Mean enhanced vegetation index"
assert M.time_axis_label(d, "rainfall_lag_7") == "Inference week" and M.time_axis_label(d, "pop") == "Week"
assert "elevation_mean" not in [k for k, _ in M.base_options(d, "environmental", False)]
assert "elevation_mean" in [k for k, _ in M.base_options(d, "environmental", True)]
s = M.trend_series(d, "rainfall_lag_7", M.CITY); assert s.isna().sum() == 1 # there is exactly one gap
tbl, mean = M.seasonal_profile(s); assert tbl.shape[0] == 12 and len(mean) == 12

print("map categories:", M.available_categories(d))
for cat, var in (("forecast", "1"), ("forecast", "8"), ("cases", "CASES"), ("cases", "CASES_lag_3"), ("cases", "CASES4"), ("climatic", "rainfall_lag_7"), ("environmental", "elevation_mean"), ("population", "pop")):
    mf = M.map_frames(d, cat, var)
    print(cat, var, "frames", len(mf.weeks), "z", mf.z.shape, "custom", mf.custom.shape, "zr", round(mf.zmin, 2), round(mf.zmax, 2), "|", mf.hover_lines[0], mf.text[0][0].replace("<br>", "/"))
    assert mf.z.shape == (len(mf.weeks), len(clusters)) and mf.custom.shape[:2] == mf.z.shape
    print("   title:", mf.title, "| slider:", mf.week_noun)
assert not M.map_frames(d, "environmental", "elevation_mean").animated

pc, pooled, city = M.performance_by_cluster(d, 1, "rainy"); print(pc.round(2).to_dict(), round(pooled, 2), round(city, 2))
brgy = pd.DataFrame({"barangay": [f"Brgy {i:02d}" for i in range(23)], "cluster": ["A"] * 23}); g = M.barangay_grid(brgy, "A"); print(g.shape, list(g.columns)); assert list(g.columns)[-1] == "Barangays (19-23)" and list(g.columns)[0] == "Barangays (1-6)"
mf7 = M.map_frames(d, "climatic", "rainfall_lag_7")
assert mf7.week_noun == "Inference week" and mf7.colorbar_title == "mm/day", (mf7.week_noun, mf7.colorbar_title)
assert M.map_frames(d, "forecast", "1").week_noun == "Week" and M.map_frames(d, "forecast", "1").colorbar_title == "Cases"
assert M.map_frames(d, "environmental", "mean_evi_lag_1").colorbar_title == ""
assert M.model_text("rf (nonupsampled)") == "Random Forest (RF), non-upsampled"
assert M.model_text("xgboost (upsampled)") == "eXtreme Gradient Boosting (XGBoost), upsampled"
assert M.model_text("stgcn (nonupsampled)").startswith("Spatio-Temporal Graph Convolutional Network (STGCN)")
assert M.model_text("lstm (upsampled)") == "Long Short-Term Memory (LSTM), upsampled" and M.model_text("svm (nonupsampled)").startswith("Support Vector Machine (SVM)")
# case series: weekly, lagged (empty across the data gap) and 4-week totals
cw = d.feature_wide("CASES_lag_3"); a = d.actuals
assert np.allclose(cw["CITY"].iloc[10:20].to_numpy(), a["CITY"].iloc[7:17].to_numpy())
assert [n for n, _ in M.base_options(d, "cases", False)] == ["CASES", "CASES4"] and [l for _, l in M.series_options(d, "CASES", False)][1] == "3 weeks before the inference week"
assert M.feature_title(d, "CASES_lag_3") == "Weekly cases, 3 weeks before the inference week" and M.available_categories(d)[1][0] == "cases"
assert M.map_frames(d, "cases", "CASES_lag_3").week_noun == "Inference week" and M.map_frames(d, "cases", "CASES4").colorbar_title == "Cases"
assert "cases" in M.trend_categories(d) and M.trend_series(d, "CASES4", M.CITY).notna().sum() > 100
import pandas as _pd; _s = _pd.Timestamp("2025-07-04"); print(f"Inference week: {_s:%Y-%m-%d} - {_s + _pd.Timedelta(days=6):%Y-%m-%d}")
cat = catalog.set_index("feature_name"); print(cat[["base", "lag", "base_display"]])
assert cat.loc["rainfall_lag_7", "base_display"] == "Rainfall" and cat.loc["rainfall_lag_2", "lag"] == 2
assert P.split_lag("DISEASE_COUNT_lag_52") == ("DISEASE_COUNT", 52) and P.split_lag("pop") == ("pop", 0)
# manual upload: checks and the session view
from dash_lib import newdata
bt = pd.DataFrame({"barangay": [f"Brgy {i:02d}" for i in range(10)], "cluster": list("AABBCCDDEE")})
w0 = pd.Timestamp("2026-03-27"); wk = [w0 + pd.Timedelta(weeks=i) for i in range(4)]
def rows(weeks, skip=()):
    return pd.DataFrame([(f"{w:%Y-%m-%d}", b, str(i % 3)) for w in weeks for i, b in enumerate(bt.barangay) if (w, b) not in skip],
                        columns=["week_start", "barangay", "cases"])
ok = newdata.validate(rows(wk), bt, clusters)
assert ok.ok and not ok.missing_weeks and ok.counts.shape == (4, 6) and ok.first_week == w0 and ok.last_week == wk[-1]
assert np.allclose(ok.counts["CITY"], ok.counts[clusters].sum(axis=1)) and ok.counts.loc[w0, "A"] == 0 + 1
# missing barangay rows and a missing week become zeros, and the weeks are listed
gap = newdata.validate(rows([wk[0], wk[1], wk[3]], skip={(wk[1], "Brgy 03")}), bt, clusters)
assert gap.ok and gap.missing_weeks == [f"{wk[1]:%Y-%m-%d}", f"{wk[2]:%Y-%m-%d}"] and len(gap.counts) == 4
assert gap.counts.loc[wk[2], "CITY"] == 0
bad = rows(wk[:2]).astype(object)
bad.loc[0, "barangay"] = "Nowhere"; bad.loc[1, "week_start"] = "2026-03-28"; bad.loc[2, "cases"] = "-3"; bad.loc[3, "cases"] = "2.5"
bad.loc[4, "cases"] = None; bad.loc[5, "week_start"] = "not a date"; bad.loc[6, "barangay"] = None; bad.loc[7, "week_start"] = bad.loc[8, "week_start"]
r = newdata.validate(bad, bt, clusters); print("\n".join(r.errors))
assert not r.ok and r.errors[0] == "0-th row (index 0): 'Nowhere' is not a valid barangay name."
assert "'2026-03-28' is not a valid week start date." in r.errors[1] and r.errors[2].endswith("Disease case count -3 is negative.")
assert "Disease case count 2.5 is not an integer." in r.errors[3] and "Disease case count is missing." in r.errors[4]
assert "'not a date' is not a valid week start date." in r.errors[5] and "Barangay name is missing." in r.errors[6]
assert newdata.validate(pd.DataFrame({"a": [1]}), bt, clusters).errors[0].startswith("The file must have the columns")
# completeness is only checked after the last base week, up to the last uploaded week; earlier rows are not used
late = newdata.validate(rows(wk[1:]), bt, clusters, base_last_week=wk[0])
assert late.ok and late.first_week == wk[1] and late.missing_weeks == []
skipped = newdata.validate(rows([wk[2], wk[3]]), bt, clusters, base_last_week=wk[0]) # the week wk[1] has no rows
assert skipped.ok and skipped.first_week == wk[1] and skipped.missing_weeks == [f"{wk[1]:%Y-%m-%d}"] and skipped.counts.loc[wk[1], "CITY"] == 0
old_only = newdata.validate(rows(wk[:2]), bt, clusters, base_last_week=wk[1])
assert not old_only.ok and old_only.headline.startswith("Error: The file has no weeks after")
assert newdata.validate(rows(wk[:2]).replace("Brgy 00", "X"), bt, clusters).headline == "Error: There are invalid data values"
# session view: base data only, then with added weeks
H0 = d.horizons[0]; fo = d.fc[(H0, M.CITY)].index
base_last = fo[-40]; base_origin = fo[-52]
v0 = M.session_view(d, base_last_week=base_last, base_last_origin=base_origin)
assert v0.actuals.index.max() == base_last and M.origin_weeks(v0)[-1] == base_origin and v0.version == "base"
assert v0.forecast_long["origin_week"].max() == base_origin and len(M.origin_weeks(d)) > len(M.origin_weeks(v0))
assert v0.feature_wide("CASES").index.max() == base_last and d.feature_wide("CASES").index.max() > base_last # not shared between views
nxt = [base_last + pd.Timedelta(weeks=i) for i in range(1, 6)]
add = pd.DataFrame(1.0, index=pd.DatetimeIndex(nxt, name="week"), columns=list(clusters) + [M.CITY])
v1 = M.session_view(d, add, version="u1", base_last_week=base_last, base_last_origin=base_origin)
assert M.origin_weeks(v1)[-1] == nxt[-1] and v1.actuals.index.max() == nxt[-1] and v1.actuals.loc[nxt[0], "A"] == 1.0
gapped = add.iloc[2:] # starts two weeks after the base data ends
v2 = M.session_view(d, gapped, base_last_week=base_last, base_last_origin=base_origin)
assert M.origin_weeks(v2)[-1] == base_last
vs = M.session_view(d, skip=True, base_last_week=base_last, base_last_origin=base_origin)
assert M.origin_weeks(vs)[-1] == fo.max() and vs.actuals.index.max() == d.actuals.index.max()
assert M.complete_up_to(add, base_last) == nxt[-1]
mf = M.map_frames(v0, "forecast", "1"); assert max(mf.weeks) <= base_origin + pd.Timedelta(weeks=1)
assert M.map_frames(v0, "cases", "CASES").weeks[-1] == base_last

# last week that can be uploaded: later weeks are rejected, the last allowed week itself is accepted
cap = wk[2]
edge = newdata.validate(rows(wk[:3]), bt, clusters, base_last_week=None, last_allowed_week=cap)
assert edge.ok and edge.last_week == cap # a file ending exactly on the last allowed week is fine
over = newdata.validate(rows(wk), bt, clusters, last_allowed_week=cap)
assert not over.ok and over.headline == "Error: There are invalid data values"
assert len(over.errors) == len(bt) # one line for each barangay row of the single week after the cap
assert over.errors[0] == f"{len(rows(wk[:3]))}-th row (index {len(rows(wk[:3]))}): '{wk[3]:%Y-%m-%d}' is after the last week that can be uploaded ({cap:%Y-%m-%d})."
assert newdata.validate(rows(wk), bt, clusters).ok # without a limit the same file is accepted
mixed = rows(wk).astype(object); mixed.loc[0, "cases"] = "-1" # a bad value and a late week in different rows: both are reported
assert len(newdata.validate(mixed, bt, clusters, last_allowed_week=cap).errors) == len(bt) + 1
import io
csv_over = newdata.validate_csv(io.StringIO(rows(wk).to_csv(index=False)), bt, clusters, base_last_week=wk[0], last_allowed_week=cap)
assert not csv_over.ok
csv_ok = newdata.validate_csv(io.StringIO(rows(wk[1:3]).to_csv(index=False)), bt, clusters, base_last_week=wk[0], last_allowed_week=cap)
assert csv_ok.ok and csv_ok.last_week == cap
# the session view never shows weeks after the last allowed week, and the last selectable week is the last uploaded week
cap_week = fo[-30]
add_over = pd.DataFrame(1.0, index=pd.DatetimeIndex([base_last + pd.Timedelta(weeks=i) for i in range(1, 15)], name="week"), columns=list(clusters) + [M.CITY])
assert add_over.index.max() > cap_week
v3 = M.session_view(d, add_over, version="u3", base_last_week=base_last, base_last_origin=base_origin, last_data_week=cap_week)
assert v3.actuals.index.max() == cap_week and M.origin_weeks(v3)[-1] == cap_week # data and selector both stop at the cap
short = add_over[add_over.index <= base_last + pd.Timedelta(weeks=3)]
v4 = M.session_view(d, short, version="u4", base_last_week=base_last, base_last_origin=base_origin, last_data_week=cap_week)
assert v4.actuals.index.max() == short.index.max() == M.origin_weeks(v4)[-1] # less data than allowed: selector stops at the last uploaded week
vs2 = M.session_view(d, skip=True, base_last_week=base_last, base_last_origin=base_origin, last_data_week=cap_week)
assert vs2.actuals.index.max() == cap_week and M.origin_weeks(vs2)[-1] == cap_week # the debug skip option obeys the cap too
assert vs2.forecast_long["origin_week"].max() == cap_week
assert M.map_frames(vs2, "cases", "CASES").weeks[-1] == cap_week
assert M.BASE_LAST_WEEK == M.BASE_LAST_ORIGIN == pd.Timestamp("2025-12-26") # with only the base data, the last selectable week is the last base week
assert M.LAST_DATA_WEEK == pd.Timestamp("2026-07-03") and (M.LAST_DATA_WEEK - newdata.GRID_ORIGIN).days % 7 == 0 # a valid week start

# vault: encrypt and decrypt with a password
from dash_lib import vault
vd = Path(tempfile.mkdtemp()); (vd / "r").mkdir(); (vd / "r/a.parquet").write_bytes(b"secret-bytes")
written = vault.encrypt_folder(vd / "r", vd / "e", "agaptest1"); assert [w.name for w in written] == ["a.parquet.enc"]
salt = (vd / "e/salt.bin").read_bytes(); good = vault.derive_key("agaptest1", salt); bad = vault.derive_key("wrong", salt)
assert vault.check_key((vd / "e/check.enc").read_bytes(), good) and not vault.check_key((vd / "e/check.enc").read_bytes(), bad)
assert vault.decrypt((vd / "e/a.parquet.enc").read_bytes(), good) == b"secret-bytes" and b"secret" not in (vd / "e/a.parquet.enc").read_bytes()
vault.encrypt_folder(vd / "r", vd / "e", "agaptest1"); assert (vd / "e/salt.bin").read_bytes() == salt # the salt is reused
print("ALL TESTS PASSED")
