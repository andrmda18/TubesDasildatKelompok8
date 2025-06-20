def show_batch_upload(model_type, task_type):
    import streamlit as st
    import pandas as pd
    import joblib

    st.title("📦 Prediksi Batch Customer")

    model_type = st.selectbox("Pilih Model", ["knn", "decisiontree", "svm", "nn"], key="model_batch")
    task_type = st.radio("Jenis Prediksi", ["klasifikasi", "regresi"], key="task_batch")

    uploaded_file = st.file_uploader("Upload File CSV", type="csv", key="file_batch")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Data Diupload:")
        st.dataframe(df)

        if st.button("Prediksi", key="predict_batch"):
            try:
                model = joblib.load(f"models/{model_type}_{task_type}.joblib")
            except FileNotFoundError:
                st.error("❌ Model tidak ditemukan.")
                return

            # Drop kolom yang tidak digunakan jika ada
            if 'id' in df.columns:
                df = df.drop(columns=['id'])

            # Encode semua kolom object menjadi angka
            for col in df.select_dtypes(include='object').columns:
                df[col] = pd.factorize(df[col])[0]

            # Cek dan urutkan kolom sesuai model
            try:
                df = df[model.feature_names_in_]
            except AttributeError:
                st.warning("⚠️ Model tidak menyimpan info fitur. Melanjutkan tanpa pengurutan.")
            except KeyError as e:
                st.error(f"⚠️ Kolom input tidak sesuai model: {e}")
                return

            try:
                prediction = model.predict(df)

                # Ubah hasil klasifikasi jadi label jika perlu
                if task_type == "klasifikasi":
                    label_mapping = {0: "gold", 1: "reguler", 2: "silver"}
                    prediction = [label_mapping.get(int(p), f"Unknown ({p})") for p in prediction]

                df["Hasil_Prediksi"] = prediction
                st.success("✅ Prediksi Berhasil!")
                st.dataframe(df)
            except Exception as e:
                st.error(f"❌ Gagal memprediksi: {e}")
