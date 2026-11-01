import streamlit as st

from dash_lib import charts, loaders, ui
from dash_lib import model as M

d = loaders.get_disease(st.session_state["disease"])
st.title(f"{d.label} Forecast Overview")

origins = M.origin_weeks(d)
week_key = f"origin_week_{d.key}"
if week_key not in st.session_state or st.session_state[week_key] not in origins:
    st.session_state[week_key] = loaders.default_origin(origins)


def _latest():
    st.session_state[week_key] = origins[-1]


with st.container(border=True):
    c1, c2 = st.columns([6, 1], vertical_alignment="bottom")
    c1.select_slider("Current week", options=list(origins), format_func=M.fmt_week, key=week_key,
                     help="All forecasts on this page were made in this week.")
    c2.button("Latest week", on_click=_latest)

origin = st.session_state[week_key]
st.markdown(f"<div style='text-align:center'><b>Forecast Overview for the week of {M.iso_span(origin)}</b></div>",
            unsafe_allow_html=True)
k = M.key_indicators(d, origin)

with st.container(border=True):
    st.markdown("**Recent past**")
    if k["past"] is None:
        st.caption("Case counts are not available in this deployment.")
    else:
        p = k["past"]
        st.caption(f"Reported cases up to the week of {M.fmt_week(origin)}")
        a, b, c = st.columns(3)
        a.metric("Cases this week", M.fmt_num(p["week_cases"], 0),
                 delta=None if p["week_change"] is None else f"{p['week_change']:+.1f}% vs last week",
                 delta_color="inverse")
        b.metric("Total cases in the past 4 weeks", M.fmt_num(p["last4"], 0),
                 delta=None if p["last4_change"] is None else f"{p['last4_change']:+.1f}% vs 4 weeks before",
                 delta_color="inverse")
        if p["top_cluster"]:
            c.metric("Most cases, past 4 weeks", M.unit_label(p["top_cluster"][0]),
                     delta=f"{p['top_cluster'][1]:,.0f} cases", delta_color="off", delta_arrow="off")

with st.container(border=True):
    st.markdown("**Forecasts made this week**")
    st.caption("Prediction intervals are reported at 80% nominal confidence.")
    if k["future"]:
        cols = st.columns(len(k["future"]))
        for col, f in zip(cols, k["future"]):
            col.metric(M.horizon_label(f["horizon"]).capitalize(), M.fmt_num(f["pred"], 1),
                       delta=f"{f['lo']:.1f} to {f['hi']:.1f}", delta_color="off", delta_arrow="off")
            col.caption(M.fmt_span_short(f["target_week"]))
    t = k["top_forecast"]
    if t:
        st.markdown(
            f"<div style='text-align:center'><b>{M.unit_label(t['cluster'])}</b> is the cluster with the highest number of cases "
            f"forecasted {M.horizon_label(t['horizon'])}:<br>{t['pred']:.1f} cases "
            f"({M.level_label(t['level'])} PI: {t['lo']:.1f} cases to {t['hi']:.1f} cases).</div>",
            unsafe_allow_html=True)


@st.fragment
def table_section():
    origin = st.session_state[week_key]
    with st.container(border=True):
        st.markdown("**Forecasts by cluster**")
        c1, c2 = st.columns([1, 3], vertical_alignment="bottom")
        h = c1.selectbox("Forecast horizon", d.horizons, format_func=M.horizon_label, key=f"tbl_h_{d.key}")
        row = M.get_forecast(d, h, M.CITY, origin)
        if row is not None:
            c2.caption(f"Week being forecasted: {M.fmt_span(row['target_week'])}")
        ui.right_aligned_table(M.cluster_forecast_table(d, origin, h))
        ui.barangay_expander(d, "ov")


@st.fragment
def bar_section():
    origin = st.session_state[week_key]
    with st.container(border=True):
        st.markdown("**Forecasts for all horizons**")
        unit = st.selectbox("Area", d.units, format_func=M.unit_label, key=f"bar_unit_{d.key}")
        bars = M.horizon_bars(d, origin, unit)
        if bars.empty:
            st.caption("No forecasts for this week.")
            return
        keys = [M.level_key(l) for l in d.levels]
        top = max(bars[f"hi{k}"].max() for k in keys)
        labels = M.bar_label_layout(bars, keys, top)
        st.plotly_chart(charts.fig_forecast_bars(bars, labels, keys, "Weekly cases"), config=charts.CHART_CONFIG)


@st.fragment
def line_section():
    origin = st.session_state[week_key]
    with st.container(border=True):
        st.markdown("**Cases and forecasts over time**")
        c = st.columns([1.3, 1.3, 1, 1, 1, 1.4], vertical_alignment="bottom")
        unit = c[0].selectbox("Area", d.units, format_func=M.unit_label, key=f"ts_unit_{d.key}")
        h = c[1].selectbox("Forecast horizon", d.horizons, format_func=M.horizon_label, key=f"ts_h_{d.key}")
        show_fc = c[2].checkbox("Forecast", True, key=f"ts_fc_{d.key}")
        show_iv = c[3].checkbox("Intervals", True, key=f"ts_iv_{d.key}")
        show_rain = c[4].checkbox("Rainy season", True, key=f"ts_rn_{d.key}")
        hide = c[5].checkbox("Hide data after current week", True, key=f"ts_hd_{d.key}")
        ts = M.timeseries_data(d, h, unit, origin, hide_future=hide)
        st.plotly_chart(charts.fig_timeseries(ts, h, "Weekly cases", show_fc, show_iv, show_rain),
                        config=charts.CHART_CONFIG, key=f"ts_{d.key}_{unit}_{h}_{origin}_{show_fc}_{show_iv}_{show_rain}_{hide}")


table_section()
bar_section()
line_section()
