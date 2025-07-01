import streamlit as st
import numpy as np
import pandas as pd
import joblib

# === Load model dan tools ===
model = joblib.load("model_svm.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
selected_features = joblib.load("selected_features.pkl")  # list nama kolom fitur terpilih (top 10)

# === Judul Aplikasi ===
st.title("🎓 Klasifikasi Topik Skripsi Mahasiswa")
st.write("Masukkan nilai mata kuliah untuk memprediksi kecenderungan topik skripsi berdasarkan kemampuan akademik.")
# Ambil fitur terpilih dari file .pkl
selected_features = joblib.load("selected_features.pkl")  # list nama kolom yang dipakai model

# Buat form input untuk setiap fitur yang dipakai model
st.subheader("📝 Masukkan Nilai Mata Kuliah")

# Simpan input user
user_input = []
for feature in selected_features:
    val = st.number_input(f"{feature}", min_value=0.0, max_value=100.0, step=0.1)
    user_input.append(val)

# Konversi ke array
if st.button("🔍 Prediksi Topik Skripsi"):
    input_df = pd.DataFrame([user_input], columns=selected_features)

    # Normalisasi
    input_scaled = scaler.transform(input_df)

    # Prediksi
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.decision_function(input_scaled)

    # Confidence
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    proba = softmax(prediction_proba)[prediction]
    predicted_label = label_encoder.inverse_transform([prediction])[0]

    # Output
    st.success(f"📌 Prediksi: **{predicted_label}**")
    st.info(f"🤖 Tingkat Keyakinan Model: **{proba * 100:.2f}%**")

