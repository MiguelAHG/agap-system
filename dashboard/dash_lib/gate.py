"""Password screen shown before anything else when the restricted data is only available encrypted."""

from __future__ import annotations

import streamlit as st

from . import loaders, vault


def unlock_screen() -> None:
    """Returns when the session is unlocked (or no lock applies); otherwise shows the password screen and stops."""
    if not loaders.lock_required() or st.session_state.get("vault_key"):
        return

    _, middle, _ = st.columns([1, 2, 1])
    with middle:
        st.header("Automated Geospatial Analysis for Disease Prediction (AGAP)")
        st.write("Welcome to the AGAP Dashboard Prototype. Please input the password.")
        with st.form("unlock"):
            password = st.text_input("Password", type="password", label_visibility="collapsed", placeholder="Password")
            submitted = st.form_submit_button("Enter")
        if submitted:
            key = vault.derive_key(password, loaders.ENC_DIR.joinpath("salt.bin").read_bytes())
            if vault.check_key(loaders.ENC_DIR.joinpath("check.enc").read_bytes(), key):
                st.session_state["vault_key"] = key
                st.rerun()
            st.error("Incorrect password.")
    st.stop()
