import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model dan alat bantu
model = joblib.load("model_svm.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
selected_features = joblib.load("selected_features.pkl")  # list nama kolom fitur yang digunakan

# Judul aplikasi
st.title("🎓 Prediksi Topik Skripsi Mahasiswa")
st.markdown("Masukkan nilai mata kuliah untuk memprediksi kecenderungan topik skripsi berdasarkan kemampuan akademik.")

# Input nilai mata kuliah
st.subheader("📝 Masukkan Nilai Mata Kuliah")
user_input = []

# Buat input sesuai urutan selected_features
for feature in selected_features:
    value = st.number_input(f"{feature}", min_value=0.0, max_value=100.0, step=0.1)
    user_input.append(value)

# Prediksi ketika tombol ditekan
if st.button("🔍 Prediksi"):
    # Konversi input jadi DataFrame
    input_df = pd.DataFrame([user_input], columns=selected_features)

    # Normalisasi (Z-score)
    input_scaled = scaler.transform(input_df)

    # Prediksi label
    prediction = model.predict(input_scaled)[0]
    decision_scores = model.decision_function(input_scaled)

    # Confidence menggunakan softmax
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    proba = softmax(decision_scores)[prediction]

    # Konversi prediksi ke label asli
    predicted_label = label_encoder.inverse_transform([prediction])[0]

    # Tampilkan hasil
    st.success(f"📌 Prediksi Topik Skripsi: **{predicted_label}**")
    st.info(f"🤖 Tingkat Keyakinan Model: **{proba * 100:.2f}%**")
