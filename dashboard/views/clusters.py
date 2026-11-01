import streamlit as st

from dash_lib import charts, loaders, ui
from dash_lib import model as M

d = loaders.get_disease(st.session_state["disease"])
st.title("Cluster Guide")

with st.container(border=True):
    st.caption("Hover over a barangay to see its cluster and district. Click Districts in the legend to show district outlines.")
    st.plotly_chart(loaders.cluster_guide_figure(), config=charts.CHART_CONFIG)
    ui.barangay_expander(d, "cg")
    ui.barangay_search_expander(d, "cg")


@st.fragment
def performance_section():
    with st.container(border=True):
        st.markdown(f"**Forecast error by cluster ({d.label})**")
        c1, c2, _ = st.columns([1, 2, 1])
        h = c1.selectbox("Forecast horizon", d.horizons, format_func=M.horizon_label, key=f"pf_h_{d.key}")
        scope_label = c2.radio("Weeks included in error computation", ["Rainy season (June to November)", "All weeks"], horizontal=True,
                               key=f"pf_s_{d.key}")
        scope = "rainy" if scope_label.startswith("Rainy") else "all"
        _, pooled, city = M.performance_by_cluster(d, h, scope)
        m1, m2, _ = st.columns([1, 1, 2])
        m1.metric("RMSE, all clusters pooled", M.fmt_num(pooled, 2))
        m2.metric("RMSE, citywide totals", M.fmt_num(city, 2))
        st.plotly_chart(loaders.performance_figure(d.key, h, scope), config=charts.CHART_CONFIG,
                        key=f"pf_{d.key}_{h}_{scope}")


if d.performance is not None:
    performance_section()
