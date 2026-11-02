"""
Loading and caching of data. Files are read once per server process with st.cache_resource, so the returned objects are
shared and must not be changed. Figures are cached too, so changing a selector reuses a figure that was already built.
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

# Put a date such as "2025-07-04" here to open the dashboard on that current week. None opens the latest week.
DEFAULT_ORIGIN_WEEK = None


def _project_root() -> Path:
    try:
        from pyprojroot import here
        return Path(here())
    except Exception:
        return Path(__file__).resolve().parents[2] # up from dash_lib to dashboard to the project root


DATA_DIR = _project_root() / "data/dashboard_data"
PUBLIC_DIR = DATA_DIR / "public"
RESTRICTED_DIR = DATA_DIR / "restricted"
ENC_DIR = DATA_DIR / "restricted_enc" # encrypted copies of the restricted files, safe to commit


def _force_lock() -> bool:
    """
    Set the environment variable AGAP_FORCE_LOCK=1 to see the password screen on a computer that has the plaintext files.
    """
    return bool(os.environ.get("AGAP_FORCE_LOCK"))


def _plaintext_path(name: str):
    path = RESTRICTED_DIR / name
    return path if path.exists() and not _force_lock() else None


def lock_required() -> bool:
    """
    True if encrypted files exist and there is no plaintext copy to use instead. This is the case for the deployed app.
    """
    has_enc = (ENC_DIR / "check.enc").exists()
    has_plain = any(RESTRICTED_DIR.glob("*.parquet")) if RESTRICTED_DIR.exists() else False
    return has_enc and (_force_lock() or not has_plain)


def read_restricted_parquet(name: str):
    """
    All restricted files are read here: true counts, forecasts, intervals and error values.
    The plaintext file is used if it exists (your own computer). If not, the encrypted file is decrypted in memory with the
    key from the password screen. If neither is available it returns None, and the app hides what needs this data.
    Change this function if the data moves somewhere else.
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
    """
    Unit text of each variable, from variable_units.json, which is written by hand. A missing file or key means no unit is shown.
    """
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
    """True if the forecast files exist, either as plaintext or encrypted."""
    return all(_plaintext_path(f"forecasts_{k}.parquet") is not None or (ENC_DIR / f"forecasts_{k}.parquet.enc").exists()
               for k in load_meta()["diseases"])


def disease_keys() -> list:
    return list(load_meta()["diseases"].keys())


def disease_label(key: str) -> str:
    return load_meta()["diseases"][key]["label"]


def skip_manual() -> bool:
    """Debug option. Uses the full disease data instead of the base data plus manual uploads."""
    return bool(st.session_state.get("skip_manual", False))


def get_disease(key: str) -> M.DiseaseData:
    """
    Returns the data for this session: the base data plus any weeks this session uploaded. See session_view in model.py.
    """
    full = _load_disease(key)
    upload = None if skip_manual() else st.session_state.get(f"new_data_{key}")
    version = "skip" if skip_manual() else (upload["version"] if upload else "base")
    cached = st.session_state.get(f"view_{key}")
    if cached is not None and cached[0] == version:
        return cached[1]
    view = M.session_view(full, None if upload is None else upload["counts"], skip_manual(), version)
    st.session_state[f"view_{key}"] = (version, view)
    return view


def warm_up() -> None:
    """Loads every disease on the first run, so switching diseases later is fast."""
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


@st.cache_resource(max_entries=8, show_spinner=False)
def map_figure(disease: str, category: str, variable: str, version: str, _d):
    """
    version names the disease data inside _d. It is part of the cache key, so a figure is not reused for different data.
    """
    frames = M.map_frames(_d, category, variable)
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
