def show_single_prediction():
    import streamlit as st
    import pandas as pd
    import joblib
    import os

    st.header("🧠 Prediksi Tunggal Customer")

    # Pilihan model dan jenis task
    st.subheader("Pilih Model dan Tipe Task")
    model_options = {
        "KNN": "knn",
        "Decision Tree": "decisiontree",
        "SVM": "svm",
        "Neural Network": "nn"
    }
    selected_models = [v for k, v in model_options.items() if st.checkbox(k)]

    task_type = st.radio("Jenis Prediksi", ["klasifikasi", "regresi"], horizontal=True)

    # Input pengguna
    st.subheader("Masukkan Data Customer")
    age = st.number_input("Usia", 18, 100, 25)
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    income = st.number_input("Pendapatan", 0, 100000, 30000)
    education = st.selectbox("Pendidikan", ["HighSchool", "College", "Bachelor", "Masters", "PhD"])
    region = st.selectbox("Wilayah", ["North", "South", "East", "West"])
    frequency = st.selectbox("Frekuensi Belanja", ["rare", "occasional", "frequent"])
    product = st.selectbox("Kategori Produk", ["Books", "Clothing", "Electronics", "Food", "Others"])
    promo = st.selectbox("Menggunakan Promo?", [0, 1])
    satisfaction = st.slider("Skor Kepuasan", 1, 10, 5)

    if task_type == "klasifikasi":
        purchase_amount = st.number_input("Jumlah Pembelian", min_value=0)

    if st.button("Prediksi"):
        if not selected_models:
            st.warning("❗ Pilih setidaknya satu model untuk melakukan prediksi.")
            return

        # Buat data input
        input_dict = {
            "age": age,
            "gender": gender,
            "income": income,
            "education": education,
            "region": region,
            "purchase_frequency": frequency,
            "product_category": product,
            "promotion_usage": promo,
            "satisfaction_score": satisfaction,
        }

        if task_type == "klasifikasi":
            input_dict["purchase_amount"] = purchase_amount

        df = pd.DataFrame([input_dict])

        # Encode kategori manual
        for col in df.select_dtypes(include='object').columns:
            df[col] = pd.factorize(df[col])[0]

        st.subheader("🔎 Hasil Prediksi dari Model Terpilih")

        for model_type in selected_models:
            model_path = f"models/{model_type}_{task_type}.joblib"

            if not os.path.exists(model_path):
                st.error(f"Model `{model_type}` tidak ditemukan.")
                continue

            try:
                model = joblib.load(model_path)
            except Exception as e:
                st.error(f"Gagal memuat model {model_type}: {e}")
                continue

            # Atur ulang kolom agar cocok dengan model
            try:
                feature_names = model.feature_names_in_
                for col in feature_names:
                    if col not in df.columns:
                        df[col] = 0  # nilai dummy jika kolom tidak tersedia

                df_model = df[feature_names]
            except AttributeError:
                df_model = df  # fallback jika model tidak punya feature_names_in_

            try:
                prediction = model.predict(df_model)[0]

                if task_type == "klasifikasi":
                    label_mapping = {
                        0: "gold",
                        1: "reguler",
                        2: "silver"
                    }
                    prediction_label = label_mapping.get(int(prediction), f"⚠️ Unknown ({prediction})")
                    st.success(f"Model **{model_type.upper()}** memprediksi: `{prediction_label}`")

                elif task_type == "regresi":
                    st.success(f"Model **{model_type.upper()}** memprediksi nilai: `{prediction:.2f}`")

                    # Tambahan penjelasan regresi
                    st.markdown("#### ℹ️ Penjelasan Prediksi")
                    if prediction < 10000:
                        st.info("🔹 Nilai prediksi termasuk **rendah**. Ini bisa berarti customer belum aktif atau potensial.")
                    elif 10000 <= prediction < 30000:
                        st.info("🟡 Nilai prediksi berada di **rentang menengah**. Customer tergolong rata-rata.")
                    else:
                        st.info("🟢 Nilai prediksi **tinggi**. Ini menunjukkan potensi customer sangat baik atau loyal.")

                    # Visualisasi
                    st.markdown("#### 📈 Visualisasi Prediksi")
                    st.bar_chart(pd.DataFrame({"Prediksi": [prediction]}))

            except Exception as e:
                st.error(f"❌ Model {model_type} gagal melakukan prediksi: {e}")
