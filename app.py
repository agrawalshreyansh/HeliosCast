import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image
from datetime import datetime
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="HeliosCast Pro", page_icon="☀️", layout="wide")

# --- SESSION STATE ---
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = "🔮 Real-Time Forecast"

# --- LOAD ASSETS ---
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'solar_model.pkl')
scaler_path = os.path.join(base_path, 'scaler.pkl')

@st.cache_resource
def load_models():
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

try:
    model, scaler = load_models()
except Exception as e:
    st.error("Error: Ensure 'solar_model.pkl' and 'scaler.pkl' are in the project folder.")
    st.stop()

# --- APP HEADER ---
st.title("☀️ HeliosCast: Solar Generation Forecasting")
st.markdown("### Professional Energy Analytics Dashboard")
st.divider()

# --- SIDEBAR NAVIGATION ---
st.sidebar.header("Navigation")
if st.sidebar.button("🔮 Real-Time Forecast", use_container_width=True):
    st.session_state.selected_tab = "🔮 Real-Time Forecast"
if st.sidebar.button("📂 Batch Processing", use_container_width=True):
    st.session_state.selected_tab = "📂 Batch Processing"
if st.sidebar.button("📊 Training & Model Metrics", use_container_width=True):
    st.session_state.selected_tab = "📊 Training & Model Metrics"

feature_names = ['shortwave_radiation', 'temperature_2m', 'cloud_cover', 'hour', 'month', 'lag_1h']

# --- TAB 1: MANUAL ENTRY ---
if st.session_state.selected_tab == "🔮 Real-Time Forecast":
    st.header("Individual Scenario Simulation")
    col_a, col_b = st.columns(2)
    with col_a:
        irradiance = st.slider("Shortwave Radiation (W/m²)", 0, 1100, 600)
        temp = st.slider("Temperature (°C)", -10, 50, 28)
        clouds = st.slider("Cloud Cover (%)", 0, 100, 15)
    with col_b:
        lag_1h = st.number_input("Last Hour Generation (Watts)", value=120.0)
        date_in = st.date_input("Forecast Date", datetime.now())
        time_in = st.time_input("Forecast Time", datetime.now())

    if st.button("Predict Power Output", use_container_width=True):
        hour, month = time_in.hour, date_in.month
        features = np.array([[irradiance, temp, clouds, hour, month, lag_1h]])
        features_scaled = scaler.transform(features)
        prediction = max(0, round(model.predict(features_scaled)[0], 4))
        
        st.metric(label="Estimated Generation", value=f"{prediction} Watts")
        st.progress(min(1.0, prediction / 1000.0))

# --- TAB 2: BATCH PROCESSING ---
elif st.session_state.selected_tab == "📂 Batch Processing":
    st.header("Bulk CSV Forecasting")
    uploaded_file = st.file_uploader("Upload weather forecast CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['month'] = df['timestamp'].dt.month
        
        if all(col in df.columns for col in feature_names):
            X_scaled = scaler.transform(df[feature_names])
            df['predicted_generation'] = model.predict(X_scaled).clip(min=0)
            st.success("Batch Prediction Complete!")
            
            # Interactive Streamlit Chart for recent data
            st.line_chart(df.set_index(df.columns[0])[['predicted_generation']].head(100))
            st.dataframe(df)
            st.download_button("Download Predictions", df.to_csv(index=False), "helios_results.csv")

# --- TAB 3: TRAINING INSIGHTS (Using your PNGs) ---
elif st.session_state.selected_tab == "📊 Training & Model Metrics":
    st.header("Model Performance & Training History")
    
    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Algorithm", "Linear Regression")
    c2.metric("R² Score", "0.8043")
    c3.metric("MAE", "0.05 Watts")

    st.divider()

    # Displaying the PNGs you generated
    col_img1, col_img2 = st.columns(2)
    
    with col_img1:
        st.subheader("1. Feature Importance")
        if os.path.exists(os.path.join(base_path, "Feature_Impact.png")):
            st.image(os.path.join(base_path, "Feature_Impact.png"), caption="How different weather factors impact power.")
        else:
            st.warning("feature_importance.png not found.")

    with col_img2:
        st.subheader("2. Model Error Distribution")
        if os.path.exists("MEA.png"):
            st.image("MEA.png", caption="Residual analysis (Errors centered at zero).")
        else:
            st.warning("residuals.png not found.")

    st.subheader("3. Actual vs Predicted Curve")
    if os.path.exists("Prediction.png"):
        st.image("Prediction.png", use_column_width=True, caption="Sample test results showing high correlation.")
    else:
        st.warning("actual_vs_pred.png not found.")

    st.subheader("4. Model Comparison")
    if os.path.exists("comparison.png"):
        st.image("comparison.png", use_column_width=True, caption="Comparison of two different models trained on the same data.")
    else:
        st.warning("comparison.png not found.")
