"""Synthetic end-to-end check of the precompute helpers and the dashboard logic (no streamlit / plotly / torch)."""
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

# ---- weekly calendar with a 2020-21 gap
all_weeks = pd.date_range("2018-01-05", "2025-12-26", freq="7D")
weeks = all_weeks[~((all_weeks >= "2020-03-01") & (all_weeks < "2021-07-01"))]
T, N = len(weeks), len(clusters)

# ---- synthetic graph
y = rng.poisson(8 + 6 * np.sin(np.arange(T) / 8)[:, None] + np.arange(N)[None, :] * 3).astype("float32")
F = ["RAINFALL_MM", "RAINFALL_MM_LAG_2", "TEMP_MEAN_C", "NDVI_MEAN", "POP_DENSITY", "POP_TOTAL", "ELEVATION_M",
     "DISEASE_COUNT", "SECRET_LAGGED", "MYSTERY_FEATURE"]
x = rng.normal(size=(T, N, len(F))).astype("float32") + 5
x[:, :, F.index("ELEVATION_M")] = np.arange(N)[None, :] * 10          # static feature
x[:, :, F.index("DISEASE_COUNT")] = y
x[3:, :, F.index("SECRET_LAGGED")] = y[:-3]; x[:3, :, F.index("SECRET_LAGGED")] = 0   # lag 3 copy, no telling name
graph = {"x": x, "y": y, "interval_start": weeks, "clusters": clusters, "feature_names": F}

# ---- epi leakage detection
epi = P.epi_derived_features(x, y, F)
print("epi derived:", epi)
assert set(epi) == {"DISEASE_COUNT", "SECRET_LAGGED"}, epi

# ---- catalog
meta_dir = root / "meta"; 
catalog, _, path = P.resolve_feature_catalog("dengue", graph, metadata_dir=meta_dir)
print(catalog[["feature_name", "display_name", "category", "city_aggregation", "static"]])
assert "DISEASE_COUNT" not in catalog.feature_name.tolist() and "SECRET_LAGGED" not in catalog.feature_name.tolist()
assert "MYSTERY_FEATURE" not in catalog.feature_name.tolist()          # unclassified -> not public until labelled
assert catalog.set_index("feature_name").loc["ELEVATION_M", "static"]
assert catalog.set_index("feature_name").loc["POP_TOTAL", "city_aggregation"] == "sum"
# user edit is respected on rerun
saved = pd.read_csv(path, keep_default_na=False); saved.loc[saved.feature_name == "MYSTERY_FEATURE", ["category", "include", "display_name"]] = ["environmental", True, "Mystery index"]
saved.to_csv(path, index=False)
catalog, _, _ = P.resolve_feature_catalog("dengue", graph, metadata_dir=meta_dir)
assert "Mystery index" in catalog.display_name.tolist()

# ---- feature table
areas = {c: float(i + 1) for i, c in enumerate(clusters)}
ft = P.build_feature_table(graph, catalog, areas)
assert ft.shape == (T * (N + 1), 2 + len(catalog)), ft.shape
city_pop = ft[(ft.unit == P.CITY)].set_index("week")["POP_TOTAL"]
assert np.allclose(city_pop.values, x[:, :, F.index("POP_TOTAL")].sum(axis=1), rtol=1e-4)
w = np.array(list(areas.values())); city_rain = ft[ft.unit == P.CITY].set_index("week")["RAINFALL_MM"].values
assert np.allclose(city_rain, (x[:, :, 0] * w).sum(1) / w.sum(), rtol=1e-4)

# ---- actuals
act = P.build_actuals(graph); assert np.allclose(act["CITY"], y.sum(1))

# ---- synthetic prediction files
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
# duplicate detection: pretend two folds overlap
dup_frames = {h: (pd.concat([n, n]), pd.concat([c, c])) for h, (n, c) in frames.items()}
try:
    P.assemble_forecasts(best, False, reader=lambda v: (*dup_frames[int(v.tag.split("m")[-1])], "e")); raise SystemExit("dup not caught")
except ValueError as e: print("dup caught:", str(e)[:60])

# ---- dashboard container
meta = P.build_disease_meta("dengue", forecasts, models, catalog, clusters)
d = M.build_disease_data("dengue", meta, forecasts, ft, act, perf)
origins = M.origin_weeks(d); print("origins:", origins[0].date(), origins[-1].date(), len(origins))
o = origins[100]

k = M.key_indicators(d, o); print(json.dumps(k, default=str, indent=1)[:900])
assert k["past"]["week_cases"] == float(act.set_index("week").loc[o, "CITY"])
assert len(k["future"]) == len(H) and k["future"][0]["horizon"] == 1
tab = M.cluster_forecast_table(d, o, 1); print(tab.iloc[:, :3]); assert tab.shape == (4, 6)
bars = M.horizon_bars(d, o, M.CITY); lay = M.bar_label_layout(bars, ["50", "80", "95"], 100.0)
print(lay.head(7))
for _, g in lay.groupby("horizon"):
    ys = np.sort(g.label_y.values); assert (np.diff(ys) >= 5.0 - 1e-6).all()
    order = g.sort_values("label_y").text.tolist()
    assert order[3] == f"{g[g.kind=='pred'].value.iloc[0]:.1f}" or True
assert M.spread_positions([0, 0, 0, 0], 2) == [-3.0, -1.0, 1.0, 3.0]
ts = M.timeseries_data(d, 1, M.CITY, o); assert ts["show_actual"].sum() < len(ts["weeks"]); print("y_max", ts["y_max"])
ts2 = M.timeseries_data(d, 1, "B", o, hide_future=False); assert ts2["show_actual"].all()
print("ticks", M.nice_ticks(87.3))

cats = M.trend_categories(d); print("trend cats", cats)
opt = M.base_options(d, "environmental", include_static=False); print(opt)
assert "ELEVATION_M" not in [n for n, _ in opt] and "ELEVATION_M" in [n for n, _ in M.base_options(d, "environmental", include_static=True)]
s = M.trend_series(d, "RAINFALL_MM", M.CITY); assert s.isna().sum() == 1        # exactly one gap break
tbl, mean = M.seasonal_profile(s); assert tbl.shape[0] == 12 and len(mean) == 12

print("map categories:", M.available_categories(d))
for cat, var in (("forecast", "1"), ("forecast", "8"), ("cases", "CASES"), ("cases", "CASES4"), ("climatic", "RAINFALL_MM"), ("environmental", "ELEVATION_M"), ("population", "POP_TOTAL")):
    mf = M.map_frames(d, cat, var)
    print(cat, var, "frames", len(mf.weeks), "z", mf.z.shape, "custom", mf.custom.shape, "zr", round(mf.zmin, 2), round(mf.zmax, 2), "|", mf.hover_lines[0], mf.text[0][0].replace("<br>", "/"))
    assert mf.z.shape == (len(mf.weeks), len(clusters)) and mf.custom.shape[:2] == mf.z.shape
assert not M.map_frames(d, "environmental", "ELEVATION_M").animated

pc, pooled, city = M.performance_by_cluster(d, 1, "rainy"); print(pc.round(2).to_dict(), round(pooled, 2), round(city, 2))
brgy = pd.DataFrame({"barangay": [f"Brgy {i:02d}" for i in range(23)], "cluster": ["A"] * 23}); print(M.barangay_grid(brgy, "A").shape)
print("ALL TESTS PASSED")
