import streamlit as st

from dash_lib import charts, loaders
from dash_lib import model as M

d = loaders.get_disease(st.session_state["disease"])
st.title(f"{d.label} Historical Trends")

labels = {k: M.CATEGORY_LABELS[k] for k in M.trend_categories(d)}


@st.fragment
def trend_section():
    if not labels:
        st.caption("No variables available.")
        return
    with st.container(border=True):
        st.markdown("**Select a historical variable and a geographic area.**")
        category = st.selectbox("Category", list(labels), format_func=labels.get, key=f"tr_cat_{d.key}")

        bases = M.base_options(d, category, include_static=False)
        _, col = st.columns([1, 7])
        base = col.selectbox("Variable", [k for k, _ in bases], format_func=dict(bases).get,
                             key=f"tr_base_{d.key}_{category}")

        series_opts = M.series_options(d, base, include_static=False)
        feature = series_opts[0][0]
        if len(series_opts) > 1:
            _, col = st.columns([2, 6])
            feature = col.selectbox("Time series", [k for k, _ in series_opts], format_func=dict(series_opts).get,
                                    key=f"tr_ser_{d.key}_{base}")

        c = st.columns([2, 2, 5], vertical_alignment="bottom")
        unit = c[0].selectbox("Area", d.units, format_func=M.unit_label, key=f"tr_unit_{d.key}")
        rain = c[1].checkbox("Rainy season", True, key=f"tr_rain_{d.key}")

    series = M.trend_series(d, feature, unit)
    name = M.variable_axis_title(d, feature)
    title = f"{M.feature_title(d, feature)}: {M.unit_label(unit)}"
    x_title = M.time_axis_label(d, feature)
    decimals = max(M.decimals_for(series.dropna().to_numpy()), 1) if category != "cases" else 0
    lag = M.feature_lag(d, feature)
    lagged = lag > 0
    with st.container(border=True):
        if lagged:
            st.caption(f"You selected a lagged variable. The value shown is from {lag} week{'s' if lag != 1 else ''} "
                       "before the inference week.")
        weekly, typical = st.tabs(["Weekly values", "Typical yearly pattern"])
        with weekly:
            st.plotly_chart(charts.fig_trend(series, name, decimals, rain, title=title, x_title=x_title),
                            config=charts.CHART_CONFIG, key=f"tr_w_{d.key}_{feature}_{unit}_{rain}")
            if rain:
                st.caption("Blue sections correspond to the June-November rainy season in each year.")
        with typical:
            by_year, mean = M.seasonal_profile(series)
            month_title = "Month of inference week" if lagged else "Month"
            st.plotly_chart(charts.fig_seasonal(by_year, mean, name, decimals, title=title, x_title=month_title),
                            config=charts.CHART_CONFIG, key=f"tr_s_{d.key}_{feature}_{unit}")


trend_section()
