import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === Load model dan tools ===
model = joblib.load("model_svm.pkl")               # Harus dilatih dengan probability=True
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
selected_features = joblib.load("selected_features.pkl")  # List 10 nama fitur

# === Judul Aplikasi ===
st.title("🎓 Prediksi Topik Skripsi Mahasiswa")
st.markdown("Masukkan nilai mata kuliah berdasarkan kemampuan akademik untuk memprediksi kecenderungan topik skripsi.")

# === Form input nilai mata kuliah ===
st.subheader("📝 Masukkan Nilai Mata Kuliah")
user_input = []

for feature in selected_features:
    val = st.number_input(f"{feature}", min_value=0, max_value=100,value=80, step=1)
    user_input.append(val)

# === Tombol prediksi ===
if st.button("🔍 Prediksi"):
    # Siapkan input data
    input_df = pd.DataFrame([user_input], columns=selected_features)

    # Normalisasi
    input_scaled = scaler.transform(input_df)

    # Prediksi label & probabilitas
    prediction = model.predict(input_scaled)[0]
    proba_all = model.predict_proba(input_scaled)[0]

    # Ambil confidence sesuai kelas hasil prediksi
    class_index = list(model.classes_).index(prediction)
    proba = proba_all[class_index]
    predicted_label = label_encoder.inverse_transform([prediction])[0]

    # === Tampilkan Hasil ===
    st.success(f"📌 Prediksi Topik Skripsi: **{predicted_label}**")
    st.info(f"🤖 Tingkat Keyakinan Model: **{proba * 100:.2f}%**")

    # === Confidence semua kelas ===
    st.subheader("📊 Confidence Semua Kelas")
    proba_df = pd.DataFrame({
        "Kelas": label_encoder.inverse_transform(model.classes_),
        "Confidence": [f"{p*100:.2f}%" for p in proba_all]
    })
    st.dataframe(proba_df.set_index("Kelas"))
