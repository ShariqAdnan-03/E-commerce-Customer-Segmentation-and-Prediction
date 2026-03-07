import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="PersonaAI", layout="wide")

# --- 2. PREMIUM CSS (Symmetry & Glassmorphism) ---
st.markdown("""
    <style>
    .main { background-color: #0B0E14; color: #FFFFFF; }
    
    /* Bento Card Styling */
    .bento-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 40px;
        text-align: center;
        transition: transform 0.3s ease;
        margin-top: 20px;
    }
    .bento-card:hover { transform: translateY(-5px); border-color: #6366f1; }
    
    /* Typography */
    .stat-label { color: #94A3B8; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 10px; }
    .stat-value { font-size: 3.5rem; font-weight: 800; margin: 0; line-height: 1; }
    .sub-text { color: #64748B; font-size: 1rem; margin-top: 10px; }
    
    /* Center aligning headers */
    .centered-header { text-align: center; margin-bottom: 30px; }
    
    /* Styling the Submit Button */
    div.stButton > button:first-child {
        background-color: #6366f1;
        color: white;
        border-radius: 12px;
        padding: 10px 40px;
        font-weight: bold;
        border: none;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        return joblib.load("customer_model.pkl"), joblib.load("segment_encoder.pkl"), joblib.load("model_features.pkl")
    except:
        return None, None, None

model, encoder, feature_cols = load_assets()

# --- 4. THE INTERFACE ---
st.markdown('<div class="centered-header"><h1>🎯 PersonaAI</h1><p style="color:#94A3B8;">Intelligent Customer Segmentation Engine</p></div>', unsafe_allow_html=True)

# Symmetrical Input Layout within a Form
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input("Recency (Days)", min_value=0, value=30)
    with col2:
        frequency = st.number_input("Frequency (Orders)", min_value=1, value=5)
    with col3:
        monetary = st.number_input("Monetary (Spend $)", min_value=0.0, value=500.0)
    
    # The Submit Button
    submit_button = st.form_submit_button("Analyze Customer")

st.markdown("<br>", unsafe_allow_html=True)

# --- 5. RESULTS LOGIC ---
if submit_button:
    if model:
        # Prediction Engine
        input_data = pd.DataFrame([[recency, frequency, monetary]], columns=feature_cols)
        prediction_num = model.predict(input_data)[0]
        prediction_label = encoder.inverse_transform([prediction_num])[0]
        confidence = np.max(model.predict_proba(input_data)[0]) * 100

        # Color Logic for the UI
        colors = {
            "VIP Segment": "#A855F7", 
            "Loyal Regulars": "#22C55E", 
            "Casual Buyers": "#EAB308", 
            "At-Risk / Churning": "#EF4444"
        }
        theme_color = colors.get(prediction_label, "#6366F1")

        # --- THE SYMMETRICAL HERO SECTION ---
        res_col1, res_col2 = st.columns(2, gap="large")
        
        with res_col1:
            st.markdown(f"""
                <div class="bento-card">
                    <p class="stat-label">Customer Persona</p>
                    <p class="stat-value" style="color: {theme_color};">{prediction_label}</p>
                    <p class="sub-text">Classification based on behavioral patterns</p>
                </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown(f"""
                <div class="bento-card">
                    <p class="stat-label">AI Confidence</p>
                    <p class="stat-value" style="color: #FFFFFF;">{confidence:.1f}%</p>
                    <p class="sub-text">Probability score from Random Forest ensemble</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.error("Model assets not found. Please ensure .pkl files are in the same directory.")

st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.caption("Engineered by Shariq Adnan | Boston Institute of Analytics Capstone 2026")