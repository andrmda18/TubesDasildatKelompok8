def show_single_prediction(model_type, task_type):
    import streamlit as st
    import pandas as pd
    import joblib

    st.header("🧠 Prediksi Tunggal Customer")

    # Input pengguna
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

        # Load model
        try:
            model = joblib.load(f"models/{model_type}_{task_type}.joblib")
        except FileNotFoundError:
            st.error("❌ Model tidak ditemukan.")
            st.stop()

        # Atur ulang kolom agar cocok dengan model
        try:
            feature_names = model.feature_names_in_
            # Tambahkan kolom dummy jika model meminta kolom yang tidak tersedia (misalnya 'id')
            for col in feature_names:
                if col not in df.columns:
                    df[col] = 0  # nilai dummy

            df = df[feature_names]
        except AttributeError:
            st.warning("⚠️ Model tidak menyimpan urutan kolom. Melanjutkan tanpa urutan eksplisit.")

        # Prediksi
        try:
            prediction = model.predict(df)[0]

            if task_type == "klasifikasi":
                label_mapping = {
                    0: "gold",
                    1: "reguler",
                    2: "silver"
                }
                prediction = label_mapping.get(int(prediction), f"⚠️ Unknown ({prediction})")

            st.success(f"Hasil Prediksi: {prediction}")
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")
