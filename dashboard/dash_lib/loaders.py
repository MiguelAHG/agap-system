"""
loaders.py

Everything that touches Streamlit caching and the disk. Data files are read once per server process with
st.cache_resource (no copy is made per rerun, so treat the returned objects as read-only). Figures that never change for
a given selection are cached the same way, so moving a selector reuses an already built chart.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from . import charts
from . import model as M
from . import vault

# Set to e.g. "2025-07-04" to open the dashboard on a specific current week. None means the latest available week.
DEFAULT_ORIGIN_WEEK = None


def _project_root() -> Path:
    try:
        from pyprojroot import here
        return Path(here())
    except Exception:
        return Path(__file__).resolve().parents[2]      # dash_lib -> dashboard -> project root


DATA_DIR = _project_root() / "data/dashboard_data"
PUBLIC_DIR = DATA_DIR / "public"
RESTRICTED_DIR = DATA_DIR / "restricted"
ENC_DIR = DATA_DIR / "restricted_enc"          # encrypted copies of the restricted files (safe to commit)


def _force_lock() -> bool:
    """Set the environment variable AGAP_FORCE_LOCK=1 to test the password screen on a machine that has the plaintext files."""
    return bool(os.environ.get("AGAP_FORCE_LOCK"))


def _plaintext_path(name: str):
    path = RESTRICTED_DIR / name
    return path if path.exists() and not _force_lock() else None


def lock_required() -> bool:
    """True when encrypted restricted data exists and there is no plaintext copy to read instead (a deployed copy)."""
    has_enc = (ENC_DIR / "check.enc").exists()
    has_plain = any(RESTRICTED_DIR.glob("*.parquet")) if RESTRICTED_DIR.exists() else False
    return has_enc and (_force_lock() or not has_plain)


def read_restricted_parquet(name: str):
    """
    The only way the app reads restricted files: true disease counts, forecasts, intervals and error metrics.
    Reads the local plaintext file when it exists (your machine), otherwise decrypts restricted_enc/<name>.enc in memory
    with the key from the password screen. Returns None when neither is available, and the app then hides everything
    that needs the restricted data. This is the single place to change if the data moves to private storage.
    """
    plain = _plaintext_path(name)
    if plain is not None:
        return pd.read_parquet(plain)
    enc, key = ENC_DIR / (name + ".enc"), st.session_state.get("vault_key")
    if enc.exists() and key:
        return pd.read_parquet(io.BytesIO(vault.decrypt(enc.read_bytes(), key)))
    return None


@st.cache_resource(show_spinner=False)
def load_meta() -> dict:
    return json.loads((PUBLIC_DIR / "meta.json").read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_units() -> dict:
    """{variable base: unit text} from variable_units.json (edited by hand). Missing file or key means no unit shown."""
    path = PUBLIC_DIR / "variable_units.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_resource(show_spinner=False)
def load_geo() -> dict:
    return {
        "clusters": json.loads((PUBLIC_DIR / "clusters.geojson").read_text(encoding="utf-8")),
        "barangays": json.loads((PUBLIC_DIR / "barangays.geojson").read_text(encoding="utf-8")),
        "extras": json.loads((PUBLIC_DIR / "geo_extras.json").read_text(encoding="utf-8")),
        "barangay_table": pd.read_parquet(PUBLIC_DIR / "barangays.parquet"),
    }


@st.cache_resource(show_spinner=False)
def _load_disease(key: str) -> M.DiseaseData:
    meta = load_meta()["diseases"][key]
    return M.build_disease_data(
        key=key,
        meta=meta,
        forecasts=read_restricted_parquet(f"forecasts_{key}.parquet"),
        features=pd.read_parquet(PUBLIC_DIR / f"features_{key}.parquet"),
        actuals=read_restricted_parquet(f"actuals_{key}.parquet"),
        performance=read_restricted_parquet(f"performance_{key}.parquet"),
        units=load_units(),
    )


def forecasts_available() -> bool:
    """True when the restricted forecast files are present (they are not in the public deployment)."""
    return all(_plaintext_path(f"forecasts_{k}.parquet") is not None or (ENC_DIR / f"forecasts_{k}.parquet.enc").exists()
               for k in load_meta()["diseases"])


def disease_keys() -> list:
    return list(load_meta()["diseases"].keys())


def disease_label(key: str) -> str:
    return load_meta()["diseases"][key]["label"]


def get_disease(key: str) -> M.DiseaseData:
    return _load_disease(key)


def warm_up() -> None:
    """Load every disease on the first run so switching diseases later is instant."""
    with st.spinner("Loading data..."):
        load_meta()
        load_geo()
        for key in disease_keys():
            _load_disease(key)


def default_origin(origins: pd.DatetimeIndex):
    if DEFAULT_ORIGIN_WEEK is not None:
        wanted = pd.Timestamp(DEFAULT_ORIGIN_WEEK)
        if wanted in origins:
            return wanted
    return origins[-1]


# ----------------------------------------------------------------------------------------------
# Cached figures (built once per selection, from small precomputed arrays)
# ----------------------------------------------------------------------------------------------

@st.cache_resource(max_entries=32, show_spinner=False)
def map_figure(disease: str, category: str, variable: str):
    d = _load_disease(disease)
    frames = M.map_frames(d, category, variable)
    geo = load_geo()
    return charts.fig_map_animated(frames, geo["clusters"], geo["extras"])


@st.cache_resource(show_spinner=False)
def cluster_guide_figure():
    geo = load_geo()
    return charts.fig_cluster_guide(geo["barangays"], geo["barangay_table"], geo["extras"])


@st.cache_resource(max_entries=32, show_spinner=False)
def performance_figure(disease: str, horizon: int, scope: str):
    d = _load_disease(disease)
    by_cluster, _, _ = M.performance_by_cluster(d, horizon, scope)
    geo = load_geo()
    return charts.fig_performance_map(by_cluster, geo["clusters"], geo["extras"])
