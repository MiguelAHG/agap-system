"""Small shared Streamlit widgets."""

from __future__ import annotations

import streamlit as st

from . import loaders
from . import model as M


def barangay_expander(d: M.DiseaseData, key_prefix: str) -> None:
    """Expander: pick a cluster, see its barangays in a compact multi-column table."""
    table = loaders.load_geo()["barangay_table"]
    with st.expander("Barangays in each cluster"):
        cluster = st.selectbox("Cluster", d.clusters, format_func=M.unit_label, key=f"{key_prefix}_cluster_{d.key}")
        grid = M.barangay_grid(table, cluster)
        n = int((table["cluster"] == cluster).sum())
        st.caption(f"{n} barangays in {M.unit_label(cluster)}")
        st.dataframe(grid, hide_index=True)


def right_aligned_table(df) -> None:
    """Show a small text table with every cell right-aligned (st.dataframe cannot right-align text cells)."""
    head = "".join(f"<th style='text-align:right;padding:6px 12px;min-width:170px;white-space:nowrap'>{c}</th>" for c in df.columns)
    rows = ""
    for name, row in df.iterrows():
        cells = "".join(f"<td style='text-align:right;padding:6px 12px;min-width:170px;white-space:nowrap'>{v}</td>" for v in row)
        rows += (f"<tr><td style='text-align:left;padding:6px 12px;font-weight:600;white-space:nowrap'>{name}</td>"
                 f"{cells}</tr>")
    st.markdown(
        "<div style='overflow-x:auto'><table style='width:100%;min-width:max-content;border-collapse:collapse;"
        "border:1px solid rgba(128,128,128,0.35)'>"
        f"<thead style='background:rgba(128,128,128,0.12)'><tr><th></th>{head}</tr></thead><tbody>{rows}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def barangay_search_expander(d: M.DiseaseData, key_prefix: str) -> None:
    """Expander: search a barangay by name, see its cluster and every barangay of that cluster."""
    table = loaders.load_geo()["barangay_table"]
    with st.expander("Search a barangay and find the cluster it belongs to"):
        names = sorted(table["barangay"].tolist())
        name = st.selectbox("Barangay", names, index=None, placeholder="Type or select a barangay",
                            key=f"{key_prefix}_search_{d.key}")
        if name is None:
            return
        cluster = table.loc[table["barangay"] == name, "cluster"].iloc[0]
        n = int((table["cluster"] == cluster).sum())
        st.markdown(f"**{name}** belongs to **{M.unit_label(cluster)}**, which has {n} barangays.")
        st.dataframe(M.barangay_grid(table, cluster), hide_index=True)
