# ==============================
# Heart Disease Classification (Random Forest)
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

# -------------------------
# Load Random Forest model
# -------------------------
model_path = "saved_models/RandomForest_best.pkl"

if not os.path.exists(model_path):
    st.error("❌ RandomForest_best.pkl not found. Please ensure it exists in 'saved_models/' folder.")
    st.stop()

model = joblib.load(model_path)
st.sidebar.success("✅ Random Forest model loaded successfully!")

# -------------------------
# App Title
# -------------------------
st.title("❤️ Heart Disease Prediction using Random Forest")
st.write("Enter the patient details below to predict the likelihood of heart disease.")

# -------------------------
# Input fields — names must exactly match training dataset columns
# -------------------------
age = st.slider("Age", 20, 90, 50)
sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
chest_pain_type = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
resting_blood_pressure = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", [1, 0])
resting_ecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
max_heart_rate = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exercise_induced_angina = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [1, 0])
st_depression = st.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0, 0.1)
st_slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
num_major_vessels = st.slider("Number of Major Vessels (0-3)", 0, 3, 0)
thalassemia = st.selectbox("Thalassemia (0=Normal, 1=Fixed defect, 2=Reversible defect, 3=Other)", [0, 1, 2, 3])

# -------------------------
# Prepare input data with exact feature names
# -------------------------
input_data = pd.DataFrame([{
    'age': age,
    'sex': sex,
    'chest_pain_type': chest_pain_type,
    'resting_blood_pressure': resting_blood_pressure,
    'cholesterol': cholesterol,
    'fasting_blood_sugar': fasting_blood_sugar,
    'resting_ecg': resting_ecg,
    'max_heart_rate': max_heart_rate,
    'exercise_induced_angina': exercise_induced_angina,
    'st_depression': st_depression,
    'st_slope': st_slope,
    'num_major_vessels': num_major_vessels,
    'thalassemia': thalassemia
}])

# -------------------------
# Prediction
# -------------------------
if st.button("🔍 Predict"):
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("📈 Prediction Result:")
        if prediction == 1:
            st.error("⚠️ **Heart Disease Detected!**")
        else:
            st.success("💚 **No Heart Disease Detected!**")

        st.write(f"**Probability of Heart Disease:** {probability:.2%}")
        st.progress(float(probability))

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

# -------------------------
# Sidebar info
# -------------------------
st.sidebar.info("🧠 Model: Random Forest Classifier trained with GridSearchCV on heart disease dataset.")
st.sidebar.markdown("---")

