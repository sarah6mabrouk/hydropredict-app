import streamlit as st
import base64
import os
import joblib as jbl
import numpy as np
import pandas as pd
import streamlit.components.v1 as components
from streamlit.components.v1 import html

# ==== CONFIG ====
st.set_page_config(page_title="HydroPredict", layout="wide")

ASSETS_DIR = "assets"
MODELS_DIR = "models"

# ==== HELPERS ====
@st.cache_data
def get_base64_image(image_path):
    """Read an image file and return base64 encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def load_base64_image(filename, label="Image"):
    """Load base64 image if file exists, else warn."""
    path = os.path.join(ASSETS_DIR, filename)
    if os.path.exists(path):
        return get_base64_image(path)
    st.warning(f"{label} not found at: {path}")
    return ""

@st.cache_resource
def load_artifacts():
    """Load ML artifacts from disk."""
    try:
        model = jbl.load(os.path.join(MODELS_DIR, "xgb_means_model.pkl"))
        scaler = jbl.load(os.path.join(MODELS_DIR, "means_scaler.pkl"))
        mean_columns = jbl.load(os.path.join(MODELS_DIR, "mean_columns.pkl"))
        default_input = jbl.load(os.path.join(MODELS_DIR, "default_input_mean.pkl"))
        return model, scaler, mean_columns, default_input
    except FileNotFoundError as e:
        st.error(f"Missing file: {e}")
        return None, None, None, None

# ==== CUSTOM CSS ====
# ==== CUSTOM CSS (inline, no external file needed) ====
CUSTOM_CSS = """
.button-link {
    display: inline-block;
    padding: 10px 20px;
    background-color: #1f77b4;
    color: white;
    border-radius: 5px;
    text-decoration: none;
}
.button-link:hover {
    background-color: #155a8a;
}
.hero-wrapper {
    position: relative;
    height: 100vh;
    color: white;
    text-align: center;
}
.hero-background {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-size: cover;
    background-position: center;
    z-index: -1;
}
.hero-content {
    position: relative;
    top: 50%;
    transform: translateY(-50%);
}
"""
st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)


# ==== GOOGLE VERIFICATION ====
if st.query_params.get("verify") == "google6b04cdb89a6ecbd5.html":
    st.write("google-site-verification: google6b04cdb89a6ecbd5.html")
    st.stop()

# ==== HEADER IMAGES ====
img_header = load_base64_image("header_pic.png", "Header Image")
img_section1 = load_base64_image("predictive maintenance.png", "Section 1 Image")

# ==== HERO SECTION ====
if img_header:
    st.markdown(f"""
    <section id="home" class="hero-wrapper">
        <div class="hero-background" style="background-image:url('data:image/jpg;base64,{img_header}')"></div>
        <div class="hero-content">
            <h1>HydroPredict - Predict Before It Fails</h1>
            <p>Built for industrial engineers, plant operators & maintenance teams</p>
            <p>HydroPredict uses advanced analytics to anticipate issues before they disrupt operations.</p>
            <a href="#predictive-maintenance" class="button-link">Learn more</a>
            <a href="https://github.com/sarah6mabrouk/Hydraulic-System-Failure-Prediction-Using-XGBoost-andPCA" target="_blank" class="button-link">GitHub</a>
        </div>
    </section>
    """, unsafe_allow_html=True)
else:
    st.error("Header image missing.")

# ==== PREDICTIVE MAINTENANCE INFO ====
st.markdown('<div id="what"></div>', unsafe_allow_html=True)
st.subheader("Not familiar with Predictive Maintenance?")
st.markdown("""
Predictive maintenance (PdM) uses historical and real-time data to monitor equipment health and anticipate failures before they happen.
""")

# ==== MODEL DEMO ====
model, scaler, mean_columns, default_input = load_artifacts()

if model and scaler and mean_columns is not None:
    col1, col2 = st.columns([2, 3])
    with col2:
        with st.form("input_form"):
            st.markdown("### Input Mean Sensor Values")
            inputs = [
                st.number_input(col, value=float(default_input[i] if default_input else 0))
                for i, col in enumerate(mean_columns)
            ]
            if st.form_submit_button("Predict"):
                X_scaled = scaler.transform(pd.DataFrame([inputs], columns=mean_columns))
                pred = model.predict(X_scaled)[0]
                label = {0: "No Leakage", 1: "Weak Leakage"}.get(pred, "Severe Leakage")
                st.success(f"Prediction: {pred} — {label}")

# ==== DASHBOARD ====
st.markdown('<div id="dashboard"></div>', unsafe_allow_html=True)
st.markdown("## 📊 Visual Insights: Dashboard")
st.markdown("""
<iframe src="https://public.tableau.com/views/InternalPumpLeakage/Dashboard1?:showVizHome=no&:embed=true" width="100%" height="900" style="border:none;"></iframe>
""", unsafe_allow_html=True)



