import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model dan preprocessing
model = joblib.load("model_svm.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
selected_features = joblib.load("selected_features.pkl")

st.set_page_config(page_title="Klasifikasi Topik Skripsi", layout="centered")

st.title("🎓 Klasifikasi Topik Skripsi Mahasiswa PTIK")
st.write("Masukkan nilai mahasiswa pada 15 fitur terpilih untuk memprediksi topik skripsi.")

# Form input nilai-nilai
input_data = []
for feature in selected_features:
    val = st.number_input(f"{feature}", min_value=0.0, max_value=100.0, value=75.0)
    input_data.append(val)

if st.button("🔍 Prediksi Topik"):
    input_array = np.array(input_data).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    pred = model.predict(scaled_input)
    label = label_encoder.inverse_transform(pred)[0]

    st.success(f"Hasil prediksi topik skripsi: **{label}** 🎯")
