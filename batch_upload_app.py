def show_batch_upload():
    import streamlit as st
    import pandas as pd
    import joblib
    import os

    st.header("📦 Prediksi Batch Customer")

    # Pilih beberapa model
    st.subheader("Pilih Model yang Ingin Dibandingkan")
    model_options = {
        "KNN": "knn",
        "Decision Tree": "decisiontree",
        "SVM": "svm",
        "Neural Network": "nn"
    }
    selected_models = [v for k, v in model_options.items() if st.checkbox(k, key=f"chk_{k}")]

    task_type = st.radio("Jenis Prediksi", ["klasifikasi", "regresi"], key="task_batch")

    uploaded_file = st.file_uploader("Upload File CSV", type="csv", key="file_batch")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("📄 Data Diupload:")
            st.dataframe(df)

            if df.empty:
                st.warning("⚠️ File kosong atau format tidak sesuai.")
                return
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            return

        if st.button("Prediksi", key="predict_batch"):
            if not selected_models:
                st.warning("❗ Pilih setidaknya satu model.")
                return

            original_df = df.copy()

            if 'id' in df.columns:
                df = df.drop(columns=['id'])

            # Encode kolom kategorikal
            for col in df.select_dtypes(include='object').columns:
                df[col] = pd.factorize(df[col])[0]

            predictions = {}

            for model_type in selected_models:
                model_path = f"models/{model_type}_{task_type}.joblib"
                if not os.path.exists(model_path):
                    st.error(f"❌ Model `{model_type}` tidak ditemukan.")
                    continue

                try:
                    model = joblib.load(model_path)
                except Exception as e:
                    st.error(f"Gagal memuat model {model_type}: {e}")
                    continue

                try:
                    df_model = df.copy()
                    if hasattr(model, 'feature_names_in_'):
                        missing_cols = [col for col in model.feature_names_in_ if col not in df_model.columns]
                        for col in missing_cols:
                            df_model[col] = 0
                        df_model = df_model[model.feature_names_in_]
                except Exception as e:
                    st.warning(f"⚠️ Gagal atur kolom untuk model {model_type}: {e}")
                    continue

                try:
                    pred = model.predict(df_model)

                    if task_type == "klasifikasi":
                        label_mapping = {0: "gold", 1: "reguler", 2: "silver"}
                        pred = [label_mapping.get(int(p), f"Unknown ({p})") for p in pred]

                    elif task_type == "regresi":
                        pred = [round(p, 2) for p in pred]

                    predictions[f"{model_type.upper()}_Prediction"] = pred

                except Exception as e:
                    st.error(f"❌ Gagal prediksi dengan model {model_type}: {e}")

            # Gabungkan hasil
            if predictions:
                result_df = original_df.copy()
                for col_name, pred_values in predictions.items():
                    result_df[col_name] = pred_values

                if task_type == "klasifikasi":
                    allowed_columns = ['age', 'income', 'loyalty_status', 'promotion_usage', 'satisfaction_score']
                    for col in predictions:
                        allowed_columns.append(col)
                    result_df = result_df[[col for col in allowed_columns if col in result_df.columns]]

                st.success("✅ Prediksi Berhasil!")
                st.markdown("#### 🧾 Tabel Hasil Prediksi")
                st.dataframe(result_df)

                if task_type == "regresi":
                    st.markdown("#### ℹ️ Interpretasi Nilai Regresi (Per Pelanggan):")
                    st.markdown("""
                    - 🔹 Nilai **< 5000** → Pelanggan dengan prediksi **rendah**, kemungkinan kurang aktif.
                    - 🟡 Nilai **5001–10000** → Pelanggan dengan potensi **menengah**.
                    - 🟢 Nilai **≥ 10001** → Pelanggan dengan prediksi **tinggi**, kemungkinan loyal.
                    """)
