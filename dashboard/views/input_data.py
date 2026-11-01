import streamlit as st

from dash_lib import loaders, newdata
from dash_lib import model as M

d = loaders.get_disease(st.session_state["disease"])
key = d.key
state_key, result_key, handled_key, counter_key = (f"new_data_{key}", f"upload_result_{key}",
                                                  f"upload_handled_{key}", f"uploader_n_{key}")
skip = loaders.skip_manual()


def _delete_upload():
    for k in (state_key, result_key, handled_key):
        st.session_state.pop(k, None)
    st.session_state[counter_key] = st.session_state.get(counter_key, 0) + 1      # empties the file uploader


def _set_skip():
    st.session_state["skip_manual"] = st.session_state["skip_widget"]


st.title("Input Disease Data")

with st.container(border=True):
    st.markdown(f"**Upload new weekly {d.label} cases**")
    st.caption("The file must be a CSV with the columns week_start, barangay and cases. There should be one row per barangay per week.\n\n"
               "To change the disease, use the selector in the sidebar.")
    if skip:
        st.info("Manual input is skipped in the debug panel below, so uploads are disabled.")

    uploaded = st.file_uploader("Upload a CSV file", type="csv", disabled=skip,
                                key=f"uploader_{key}_{st.session_state.get(counter_key, 0)}")
    if uploaded is not None and uploaded.file_id != st.session_state.get(handled_key):
        st.session_state[handled_key] = uploaded.file_id
        st.session_state.pop(state_key, None)                  # an earlier upload is treated as if it never happened
        result = newdata.validate_csv(uploaded, loaders.load_geo()["barangay_table"], d.clusters, M.CITY,
                                        base_last_week=M.BASE_LAST_WEEK)
        if result.ok:
            st.session_state[state_key] = {"counts": result.counts, "version": result.version, "name": uploaded.name}
        st.session_state[result_key] = {"ok": result.ok, "errors": result.errors, "headline": result.headline,
                                        "missing_weeks": result.missing_weeks}
        d = loaders.get_disease(key)

    result = st.session_state.get(result_key)
    if result is not None and not skip:
        if result["ok"]:
            if result["missing_weeks"]:
                st.warning("Warning: In these weeks, rows for some barangays are missing. "
                           "Disease counts for missing barangays were treated as zero:")
                with st.expander("Log of weeks with missing barangays"):
                    st.text("\n".join(result["missing_weeks"]))
            latest = M.complete_up_to(d.actuals, M.BASE_LAST_WEEK)
            st.info(f"New data has been added. You can now select weeks up to {M.fmt_week(latest)}.")
        else:
            st.error(result["headline"])
            with st.expander("Log of invalid values"):
                lines = result["errors"]
                st.text("\n".join(lines[:500]) + (f"\n... and {len(lines) - 500} more rows with errors." if len(lines) > 500 else ""))

    if st.session_state.get(state_key) is not None and not skip:
        st.button("Delete uploaded data", on_click=_delete_upload)

st.warning("This is a prototype feature for demonstration purposes only. "
           "Your upload will be deleted if you reuse the upload button or close the app.")

with st.expander("Debug panel"):
    st.write("In this dashboard prototype, the user must manually input disease data from April 2026 onward, "
             "so that the prototype can demonstrate the data input mechanism. "
             "For debugging purposes, this manual input can be skipped.")
    st.checkbox("Skip manual input for 2026 April-September", value=skip, key="skip_widget", on_change=_set_skip)
