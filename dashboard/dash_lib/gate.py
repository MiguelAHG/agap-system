"""Password screen. It is shown first when the restricted data exists only in encrypted form."""

from __future__ import annotations

import streamlit as st

from . import loaders, vault


def unlock_screen() -> None:
    """
    Does nothing if the session is already unlocked or no lock is needed. Otherwise it shows the password form and stops the page.
    """
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
