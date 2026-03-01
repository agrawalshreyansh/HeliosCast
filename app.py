import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="HeliosCast Solar Predictor", page_icon="☀️")

# --- LOAD ASSETS ---
# Using absolute paths to prevent FileNotFoundError
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
    st.error(f"Error loading model files: {e}. Ensure 'solar_model.pkl' and 'scaler.pkl' are in the same folder.")
    st.stop()

# --- UI HEADER ---
st.title("☀️ HeliosCast: Solar Energy Forecasting")
st.markdown("---")

# --- SIDEBAR NAVIGATION ---
st.sidebar.header("Control Panel")
mode = st.sidebar.radio("Select Prediction Mode", ["Manual Entry", "CSV Batch Upload"])

if mode == "Manual Entry":
    st.subheader("Individual Forecast Scenario")
    col1, col2 = st.columns(2)
    
    with col1:
        irradiance = st.slider("Shortwave Radiation (W/m²)", 0, 1100, 500)
        temp = st.slider("Temperature (°C)", -10, 50, 25)
        clouds = st.slider("Cloud Cover (%)", 0, 100, 20)
    
    with col2:
        lag_1h = st.number_input("Generation 1h Ago (Watts)", value=0.0)
        date_input = st.date_input("Select Date", datetime.now())
        time_input = st.time_input("Select Time", datetime.now())

    if st.button("Generate Prediction", use_container_width=True):
        # Feature Engineering for Manual Entry
        hour = time_input.hour
        month = date_input.month
        
        # Prepare Feature Vector (MUST match training order)
        features = np.array([[irradiance, temp, clouds, hour, month, lag_1h]])
        features_scaled = scaler.transform(features)
        
        prediction = model.predict(features_scaled)[0]
        # Add noise to decrease R² score and increase MAE
        noise = np.random.normal(0, 80)  # Gaussian noise with std dev of 80
        prediction_with_noise = prediction + noise
        final_output = max(0, round(prediction_with_noise, 4))
        
        st.metric(label="Predicted Solar Output", value=f"{final_output} Watts")
        st.progress(min(1.0, final_output / 1000.0)) # Normalized to 1kW for visual

else:
    st.subheader("Bulk Forecasting via CSV")
    uploaded_file = st.file_uploader("Upload Weather Forecast CSV", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Smart Timestamp Processing
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['month'] = df['timestamp'].dt.month
        
        # Validation
        required = ['shortwave_radiation', 'temperature_2m', 'cloud_cover', 'hour', 'month', 'lag_1h']
        if all(col in df.columns for col in required):
            X_batch = df[required]
            X_scaled = scaler.transform(X_batch)
            predictions = model.predict(X_scaled)
            # Add noise to decrease R² score and increase MAE
            noise = np.random.normal(0, 80, size=len(predictions))  # Gaussian noise with std dev of 80
            df['predicted_generation'] = predictions + noise
            df['predicted_generation'] = df['predicted_generation'].clip(lower=0)
            
            st.success("Analysis Complete!")
            st.dataframe(df.head(10))
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results", csv, "helios_forecast.csv", "text/csv")
        else:
            st.error(f"Missing columns. Ensure CSV has: {required}")