import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json

# ------------------------------
# 1️⃣ Load Model, Scaler, and Metadata
# ------------------------------
st.title("🏭 Manufacturing Output Prediction App")

try:
    model = joblib.load("model_artifacts/ridge_model.pkl")
    scaler = joblib.load("model_artifacts/scaler.pkl")

    with open("model_artifacts/model_metadata.json", "r") as f:
        metadata = json.load(f)

    st.sidebar.success("✅ Model and Scaler Loaded Successfully")
except Exception as e:
    st.error(f"❌ Error loading model or scaler: {e}")
    st.stop()

# ------------------------------
# 2️⃣ Input Parameters (Sidebar)
# ------------------------------
st.sidebar.header("🔧 Input Parameters")

cycle_time = st.sidebar.number_input("Cycle Time (s)", value=50.0)
temperature = st.sidebar.number_input("Injection Temperature (°C)", value=200.0)
pressure = st.sidebar.number_input("Injection Pressure (bar)", value=80.0)
cooling_time = st.sidebar.number_input("Cooling Time (s)", value=10.0)
material_viscosity = st.sidebar.number_input("Material Viscosity", value=1.2)
ambient_temp = st.sidebar.number_input("Ambient Temperature (°C)", value=25.0)
machine_age = st.sidebar.number_input("Machine Age (years)", value=5.0)
operator_exp = st.sidebar.number_input("Operator Experience (years)", value=3.0)
maintenance_hours = st.sidebar.number_input("Maintenance Hours per Month", value=20.0)

# 🔹 Add derived feature — Efficiency Ratio
efficiency_ratio = st.sidebar.number_input(
    "Efficiency Ratio (Parts per Hour / Cycle Time)", value=0.9
)

# ------------------------------
# 3️⃣ Prepare Input Data
# ------------------------------
feature_names = [
    'Injection_Temperature',
    'Injection_Pressure',
    'Cycle_Time',
    'Cooling_Time',
    'Material_Viscosity',
    'Ambient_Temperature',
    'Machine_Age',
    'Operator_Experience',
    'Maintenance_Hours',
    'Efficiency_Ratio'
]

input_features = [
    temperature,
    pressure,
    cycle_time,
    cooling_time,
    material_viscosity,
    ambient_temp,
    machine_age,
    operator_exp,
    maintenance_hours,
    efficiency_ratio
]

# Create DataFrame with feature names
input_df = pd.DataFrame([input_features], columns=feature_names)

# Scale the input using saved scaler
try:
    input_scaled = scaler.transform(input_df)
except Exception as e:
    st.error(f"⚠️ Error scaling input: {e}")
    st.stop()

# ------------------------------
# 4️⃣ Make Prediction
# ------------------------------
try:
    predicted_parts = model.predict(input_scaled)[0]
    st.subheader("📈 Predicted Output")
    st.success(f"Predicted Parts per Hour: **{predicted_parts:.2f}**")
except Exception as e:
    st.error(f"❌ Prediction Error: {e}")

# ------------------------------
# 5️⃣ Model Metadata Display
# ------------------------------
st.sidebar.subheader("📊 Model Info")
try:
    st.sidebar.json(metadata)
except:
    st.sidebar.warning("⚠️ Metadata file not found or invalid format.")
