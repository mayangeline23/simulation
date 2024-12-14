import streamlit as st
import pandas as pd
from io import BytesIO

# Unified File Upload Section
st.sidebar.header("Upload Dataset File")

# Section for Import Data for Display
uploaded_file_display = st.sidebar.file_uploader("Import Data for Display (CSV or Excel)", type=["csv", "xlsx"], key="display")
data_display = None

if uploaded_file_display is not None:
    # Process CSV files for display
    if uploaded_file_display.name.endswith(".csv"):
        data_display = pd.read_csv(uploaded_file_display)
        st.write("### Uploaded CSV File for Display:")
        st.dataframe(data_display)

    # Process Excel files for display
    elif uploaded_file_display.name.endswith(".xlsx"):
        data_display = pd.read_excel(uploaded_file_display)
        st.write("### Uploaded Excel File for Display:")
        st.dataframe(data_display)

# Section for Import Data for Download
uploaded_file_download = st.sidebar.file_uploader("Import Data for Download (CSV or Excel)", type=["csv", "xlsx"], key="download")
data_download = None

if uploaded_file_download is not None:
    # Process CSV files for download
    if uploaded_file_download.name.endswith(".csv"):
        data_download = pd.read_csv(uploaded_file_download)

    # Process Excel files for download
    elif uploaded_file_download.name.endswith(".xlsx"):
        data_download = pd.read_excel(uploaded_file_download)

    # Prepare CSV for download
    csv_buffer = BytesIO()
    data_download.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue()

    # Prepare Excel for download
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        data_download.to_excel(writer, index=False, sheet_name='Sheet1')
    excel_bytes = excel_buffer.getvalue()

    st.sidebar.write("Download options for the uploaded file:")

    # CSV download button
    st.sidebar.download_button(
        label="Download as CSV",
        data=csv_bytes,
        file_name="data.csv",
        mime="text/csv",
    )

    # Excel download button
    st.sidebar.download_button(
        label="Download as Excel",
        data=excel_bytes,
        file_name="data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

else:
    st.sidebar.write("Upload a file to enable display and download options.")