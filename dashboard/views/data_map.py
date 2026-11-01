import streamlit as st

from dash_lib import charts, loaders
from dash_lib import model as M

d = loaders.get_disease(st.session_state["disease"])
st.title(f"{d.label} Data Map")


@st.fragment
def map_section():
    cats = M.available_categories(d)
    # Selectors are stacked and stepped to the right: category, then variable, then time series.
    category = st.selectbox("Category", [k for k, _ in cats], format_func=dict(cats).get, key=f"map_cat_{d.key}")

    options = M.variable_options(d, category)
    _, col = st.columns([1, 7])
    variable = col.selectbox("Variable", [k for k, _ in options], format_func=dict(options).get,
                             key=f"map_var_{d.key}_{category}")

    lag = 0
    if category in M.TREND_CATEGORIES:
        series_opts = M.series_options(d, variable, include_static=True)
        series_key = f"map_ser_{d.key}_{variable}"
        variable = series_opts[0][0]
        if len(series_opts) > 1:
            _, col = st.columns([2, 6])
            variable = col.selectbox("Time series", [k for k, _ in series_opts], format_func=dict(series_opts).get,
                                     key=series_key)
        lag = M.feature_lag(d, variable)

    if category == "forecast":
        st.markdown(
            f"<u>Note.</u> The best model for forecasting {M.horizon_label(int(variable))} was used, "
            f"**{M.model_text(d.models[int(variable)])}.**", unsafe_allow_html=True)
    elif lag > 0:
        st.markdown(
            "<u>Note.</u> Each week shown is the inference week. "
            f"The values are lagged and come from {lag} week{'s' if lag != 1 else ''} prior.", unsafe_allow_html=True)

    with st.container(border=True):
        st.plotly_chart(loaders.map_figure(d.key, category, variable), key=f"map_{d.key}_{category}_{variable}",
                        config=charts.CHART_CONFIG)

    with st.expander("How to use this map?"):
        st.markdown(
            "Letters A, B, C, etc. refer to Barangay Cluster names. To learn more about clusters, see the Cluster Guide.\n\n"
            "Below the map, press Play or drag the slider to change the week.")


map_section()
