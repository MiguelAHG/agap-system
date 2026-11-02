"""Starting point of the dashboard"""

import streamlit as st

from dash_lib import gate, loaders

st.set_page_config(page_title="AGAP Disease Forecasting", layout="wide", initial_sidebar_state="expanded")

gate.unlock_screen()

loaders.warm_up()

pages = [
    st.Page("views/data_map.py", title="Data Map", default=not loaders.forecasts_available()),
    st.Page("views/trends.py", title="Historical Trends"),
    st.Page("views/clusters.py", title="Cluster Guide"),
]
if loaders.forecasts_available():
    pages.insert(0, st.Page("views/overview.py", title="Forecast Overview", default=True))
    pages.append(st.Page("views/input_data.py", title="Input Disease Data"))
navigation = st.navigation(pages, position="sidebar")

with st.sidebar:
    st.radio("Disease", loaders.disease_keys(), format_func=loaders.disease_label, key="disease")

navigation.run()
