import streamlit as st
from single_prediction_app import show_single_prediction
from batch_upload_app import show_batch_upload

st.set_page_config(page_title="Customer Prediction App", layout="wide")

st.title("📊 Aplikasi Prediksi Customer")

# Sidebar
st.sidebar.title("🔧 Pengaturan")
page = st.sidebar.radio("Pilih Mode", ["Single Prediction", "Batch Upload"])
model_type = st.sidebar.selectbox("Pilih Model", ["knn", "decisiontree", "svm", "nn"])
task_type = st.sidebar.radio("Jenis Prediksi", ["klasifikasi", "regresi"])

# Tampilkan halaman sesuai pilihan
if page == "Single Prediction":
    show_single_prediction(model_type, task_type)
elif page == "Batch Upload":
    show_batch_upload(model_type, task_type)
