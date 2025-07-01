import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model dan tools
model = joblib.load("model_svm.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
selected_features = joblib.load("selected_features.pkl")  # list of feature names

# Judul
st.title("🎓 Prediksi Topik Skripsi Mahasiswa")
st.markdown("Masukkan nilai mata kuliah berdasarkan kemampuan akademik untuk memprediksi kecenderungan topik skripsi.")

# Buat form input fitur
st.subheader("📝 Input Nilai Mata Kuliah")
input_values = {}

# Ambil input dari user berdasarkan selected_features
for feature in selected_features:
    input_values[feature] = st.number_input(f"{feature}", min_value=0.0, max_value=100.0, step=0.1)

# Jika tombol ditekan
if st.button("🔍 Prediksi"):
    # Buat DataFrame 1 baris dengan nama kolom sesuai
    input_df = pd.DataFrame([input_values], columns=selected_features)

    # Normalisasi Z-score
    input_scaled = scaler.transform(input_df)

    # Prediksi
    prediction = model.predict(input_scaled)[0]
    decision_scores = model.decision_function(input_scaled)

    # Confidence dari decision_function + softmax
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    proba = softmax(decision_scores)[prediction]
    predicted_label = label_encoder.inverse_transform([prediction])[0]

    # Output hasil
    st.success(f"📌 Prediksi: **{predicted_label}**")
    st.info(f"🤖 Tingkat Keyakinan Model: **{proba * 100:.2f}%**")
