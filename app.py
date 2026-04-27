import streamlit as st
import pandas as pd
import numpy as np
import os

# Set page config for aesthetics
st.set_page_config(
    page_title="Logistics Delay Predictor",
    page_icon="🚚",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    /* Gradient background and modern typography */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    h1 {
        color: #38bdf8;
        font-weight: 800;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    h2, h3 {
        color: #94a3b8;
    }
    
    /* Glassmorphism cards */
    div[data-testid="stForm"] {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 1rem;
        padding: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.2);
    }
    
    /* Inputs */
    .stTextInput input, .stSelectbox select, .stNumberInput input {
        background-color: #334155;
        color: white;
        border-radius: 0.5rem;
        border: 1px solid #475569;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(to right, #3b82f6, #06b6d4);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.5);
    }
    
    /* Result card */
    .result-card {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid #10b981;
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
        margin-top: 2rem;
        animation: fadeIn 0.5s ease-out;
    }
    .result-value {
        font-size: 3.5rem;
        font-weight: 800;
        color: #10b981;
        text-shadow: 0 0 10px rgba(16, 185, 129, 0.3);
    }
    .result-label {
        font-size: 1.2rem;
        color: #a7f3d0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

st.title("🚚 Predictive Logistics Delay Pipeline")
st.markdown("Enter real-world logistics parameters to predict estimated delays.")

# Initialize MLflow with secrets securely
def load_mlflow_model():
    try:
        import mlflow
        # Safely access st.secrets
        if "mlflow" in st.secrets:
            uri = st.secrets["mlflow"].get("MLFLOW_TRACKING_URI", "")
            user = st.secrets["mlflow"].get("MLFLOW_TRACKING_USERNAME", "")
            password = st.secrets["mlflow"].get("MLFLOW_TRACKING_PASSWORD", "")
            
            if uri:
                os.environ["MLFLOW_TRACKING_URI"] = uri
                mlflow.set_tracking_uri(uri)
            if user:
                os.environ["MLFLOW_TRACKING_USERNAME"] = user
            if password:
                os.environ["MLFLOW_TRACKING_PASSWORD"] = password
                
        # Attempt to load model
        st.sidebar.info("Attempting to connect to DagsHub MLflow registry...")
        model = mlflow.pyfunc.load_model("models:/predictive_logistics_model/latest")
        st.sidebar.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.sidebar.warning(f"Could not connect to MLflow (using placeholder secrets). Using fallback prediction for UI demonstration.\n\nError: {e}")
        return None

model = load_mlflow_model()

# Form inputs
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        origin = st.selectbox("Origin", ["Mumbai", "Bengaluru", "Pune", "Hyderabad"], index=0)
        distance = st.number_input("Distance (km)", min_value=1.0, max_value=5000.0, value=150.5, step=1.0)
        weather_code = st.selectbox("Current Weather", ["Clear", "Cloudy", "Rain", "Heavy Rain", "Storm"], index=0)
        
    with col2:
        destination = st.selectbox("Destination", ["Bengaluru", "Mumbai", "Pune", "Hyderabad"], index=0)
        traffic_density = st.slider("Traffic Density (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
        priority = st.selectbox("Priority", ["Standard", "Express", "Critical"], index=0)
        
    submit_button = st.form_submit_button("Predict Delay")

if submit_button:
    with st.spinner("Analyzing logistics network and running prediction..."):
        # Map inputs to dummy dataframe mimicking processed features
        # In a real app, we would run `engineer.py` transformers here.
        weather_map = {"Clear": 1, "Cloudy": 3, "Rain": 6, "Heavy Rain": 8, "Storm": 10}
        
        df = pd.DataFrame({
            "trip_distance": [distance],
            "fare_amount": [distance * 1.5], # dummy mapping
            "weather_code": [weather_map[weather_code]],
            "traffic_index": [traffic_density],
            "priority": [priority], # Used for UI, possibly ignored by model
            "origin": [origin],
            "destination": [destination]
        })
        
        # Predict
        predicted_delay = 0.0
        if model is not None:
            try:
                # Try passing raw-ish df, it might fail if model expects exact feature cols
                pred = model.predict(df)
                predicted_delay = float(pred[0])
            except Exception as e:
                st.error(f"Prediction failed with schema error (expected processed features): {e}")
                # Fallback calculation
                predicted_delay = distance * 0.05 + traffic_density * 45 + weather_map[weather_code] * 5
        else:
            # Fallback calculation for UI demo
            predicted_delay = distance * 0.05 + traffic_density * 45 + weather_map[weather_code] * 5
            
        # Display results with beautiful UI
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Predicted Delay</div>
            <div class="result-value">{predicted_delay:.1f} <span style="font-size:1.5rem; color:#6ee7b7;">hours</span></div>
            <div style="color: #94a3b8; margin-top: 1rem; font-size: 0.9rem;">
                Based on current weather conditions in {origin} and traffic density.
            </div>
        </div>
        """, unsafe_allow_html=True)
