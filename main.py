import streamlit as st
from single_prediction_app import show_single_prediction
from batch_upload_app import show_batch_upload

st.set_page_config(page_title="Customer Prediction App", layout="wide")

st.title("📊 Aplikasi Prediksi Customer")

# Sidebar hanya untuk memilih mode
st.sidebar.title("🔧 Pengaturan")
page = st.sidebar.radio("Pilih Mode", ["Single Prediction", "Batch Upload"])

# Tampilkan halaman sesuai pilihan
if page == "Single Prediction":
    show_single_prediction()
elif page == "Batch Upload":
    show_batch_upload()
