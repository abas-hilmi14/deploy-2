import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model dan alat bantu
model = joblib.load("model_svm.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
selected_features = joblib.load("selected_features.pkl")  # list of 10 features

# Judul
st.title("🎓 Prediksi Topik Skripsi Mahasiswa")
st.markdown("Masukkan nilai mata kuliah untuk memprediksi topik skripsi berdasarkan kemampuan akademik.")

# Ambil input user
st.subheader("📝 Input Nilai Mata Kuliah")
user_input = []

for feature in selected_features:
    val = st.number_input(f"{feature}", min_value=0.0, max_value=100.0, step=0.1)
    user_input.append(val)

if st.button("🔍 Prediksi"):
    # Konversi ke DataFrame (1 baris, kolom = selected_features)
    input_df = pd.DataFrame([user_input], columns=selected_features)

    # Normalisasi
    input_scaled = scaler.transform(input_df)

    # Debug: tampilkan bentuk input dan ekspektasi model
    st.write("📊 Shape input_scaled:", input_scaled.shape)
    st.write("📊 Expected features by model:", model.n_features_in_)

    # Validasi jumlah fitur
    if input_scaled.shape[1] != model.n_features_in_:
        st.error(f"❌ Jumlah fitur tidak sesuai. Model mengharapkan {model.n_features_in_} fitur, tetapi Anda memasukkan {input_scaled.shape[1]}.")
    else:
        # Prediksi label
        prediction = model.predict(input_scaled)[0]
        decision_scores = model.decision_function(input_scaled)
        
        # Softmax helper
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()
        
        # Ambil confidence berdasarkan tipe klasifikasi
        if len(model.classes_) == 2:
            # Binary classification → decision_function hasil 1 dimensi
            proba = float(1 / (1 + np.exp(-decision_scores[0])))  # sigmoid
            predicted_index = list(model.classes_).index(prediction)
            if predicted_index == 0:
                proba = 1 - proba  # ambil probabilitas untuk prediksi kelas 0
        else:
            # Multiclass → pakai softmax
            class_index = list(model.classes_).index(prediction)
            proba = softmax(decision_scores)[class_index]
        
                
        # Konversi ke label asli
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        
        # Output
        st.success(f"📌 Prediksi: **{predicted_label}**")
        st.info(f"🤖 Tingkat Keyakinan Model: **{proba * 100:.2f}%**")

