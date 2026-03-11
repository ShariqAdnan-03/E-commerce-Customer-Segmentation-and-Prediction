import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Customer Segmentation & Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Manrope:wght@400;500;600;700&display=swap');

/* Hide Streamlit chrome */
#MainMenu, footer, header, .stDeployButton,
[data-testid="stStatusWidget"], [data-testid="stToolbar"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }

html, body { margin: 0 !important; padding: 0 !important; overflow: hidden !important; }
.stApp { background: #FFFFFF !important; height: 100vh !important; overflow: hidden !important; }

.block-container {
    padding: 2.2rem 6rem 1rem 6rem !important;
    max-width: 100% !important;
    overflow: hidden !important;
}

body, p, span, div, label { font-family: 'Manrope', sans-serif !important; }

/* ── Eyebrow tag ── */
.eyebrow {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #EA580C;
    background: #FFF7ED;
    border: 1px solid #FDBA74;
    padding: 5px 14px;
    border-radius: 100px;
    margin-bottom: 14px;
}

/* ── Title ── */
.main-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 3.6rem !important;
    font-weight: 800 !important;
    line-height: 1.05 !important;
    color: #09090B !important;
    margin: 0 0 14px 0 !important;
    letter-spacing: -2px !important;
}

/* ── Subtitle ── */
.subtitle {
    font-size: 0.95rem;
    color: #71717A;
    line-height: 1.6;
    margin-bottom: 14px;
    font-weight: 400;
    max-width: 680px;
}

/* ── Orange accent line ── */
.accent-line {
    width: 44px; height: 3px;
    background: #F97316;
    border-radius: 2px;
    margin-bottom: 18px;
}

/* ── RFM Info Cards ── */
.rfm-grid { display: flex; gap: 14px; margin-bottom: 18px; }
.rfm-item {
    flex: 1;
    background: #FAFAFA;
    border: 1px solid #EFEFEF;
    border-radius: 12px;
    padding: 14px 16px;
}
.rfm-icon { font-size: 1.1rem; margin-bottom: 5px; display: block; }
.rfm-name {
    display: block;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.82rem;
    font-weight: 700;
    color: #18181B;
    margin-bottom: 3px;
}
.rfm-desc { font-size: 0.73rem; color: #A1A1AA; margin: 0; line-height: 1.45; }

/* ── Form ── */
div[data-testid="stForm"] {
    background: #FAFAFA !important;
    border: 1px solid #EFEFEF !important;
    border-radius: 16px !important;
    padding: 22px 24px !important;
}
div[data-testid="stForm"] > div:first-child { padding: 0 !important; }

/* ── Number Inputs ── */
div[data-testid="stNumberInput"] input {
    background: #FFFFFF !important;
    border: 1.5px solid #E4E4E7 !important;
    border-radius: 10px !important;
    color: #09090B !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    padding: 10px 14px !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: #F97316 !important;
    box-shadow: 0 0 0 3px rgba(249,115,22,0.12) !important;
    outline: none !important;
}
div[data-testid="stNumberInput"] label {
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.8px !important;
    color: #52525B !important;
}

/* ── Submit Button ── */
div[data-testid="stFormSubmitButton"] > button {
    background: #09090B !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    padding: 13px 36px !important;
    width: 100% !important;
    margin-top: 10px !important;
    letter-spacing: 0.3px !important;
    transition: background 0.2s ease !important;
    cursor: pointer !important;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    background: #F97316 !important;
}

/* ── Result Cards ── */
.result-row { display: flex; gap: 16px; margin-top: 18px; }

.result-card {
    flex: 1;
    background: #09090B;
    border-radius: 14px;
    padding: 20px 24px;
}
.result-label {
    display: block;
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2.2px;
    color: #71717A;
    margin-bottom: 6px;
}
.result-value {
    display: block;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.6rem;
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 4px;
}
.result-sub {
    display: block;
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.72rem;
    color: #71717A;
    font-weight: 400;
}

/* ── Footer ── */
.footer-text {
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.7rem;
    color: #D4D4D8;
    text-align: center;
    padding-top: 14px;
    font-weight: 500;
    letter-spacing: 0.3px;
}

/* Column gap fix */
div[data-testid="stHorizontalBlock"] { gap: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    try:
        return (
            joblib.load("customer_model.pkl"),
            joblib.load("segment_encoder.pkl"),
            joblib.load("model_features.pkl")
        )
    except:
        return None, None, None

model, encoder, feature_cols = load_assets()

# ── HERO ───────────────────────────────────────────────────────────────────────
st.markdown("""
<span class="eyebrow">&#x26A1; RFM Analysis Engine</span>
<h1 class="main-title">E-Commerce Customer Segmentation & Prediction</h1>
<p class="subtitle">
    Enter a customer's behavioural data below. The model instantly classifies them
    into one of four actionable segments using a trained Random Forest ensemble.
</p>
<div class="accent-line"></div>
""", unsafe_allow_html=True)

# ── RFM INFO CARDS ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="rfm-grid">
  <div class="rfm-item">
    <span class="rfm-icon">&#x1F550;</span>
    <span class="rfm-name">Recency</span>
    <p class="rfm-desc">Days since the customer's last purchase. A lower value means they shopped more recently and are likely still engaged.</p>
  </div>
  <div class="rfm-item">
    <span class="rfm-icon">&#x1F4E6;</span>
    <span class="rfm-name">Frequency</span>
    <p class="rfm-desc">Total number of orders placed. Higher frequency signals a strong and consistent purchase habit over time.</p>
  </div>
  <div class="rfm-item">
    <span class="rfm-icon">&#x1F4B0;</span>
    <span class="rfm-name">Monetary</span>
    <p class="rfm-desc">Total amount spent in dollars. A higher value indicates greater lifetime value and revenue contribution.</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── INPUT FORM ─────────────────────────────────────────────────────────────────
with st.form("rfm_form"):
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        recency = st.number_input("Recency (Days)", min_value=0, value=30, step=1)
    with c2:
        frequency = st.number_input("Frequency (Orders)", min_value=1, value=5, step=1)
    with c3:
        monetary = st.number_input("Monetary ($)", min_value=0.0, value=500.0, step=10.0)
    submitted = st.form_submit_button("Analyze Customer →")

# ── RESULTS ────────────────────────────────────────────────────────────────────
if submitted:
    if model and encoder and feature_cols:
        df_in      = pd.DataFrame([[recency, frequency, monetary]], columns=feature_cols)
        pred_num   = model.predict(df_in)[0]
        pred_label = encoder.inverse_transform([pred_num])[0]
        conf       = np.max(model.predict_proba(df_in)[0]) * 100

        colors = {
            "VIP Segment":        "#A855F7",
            "Loyal Regulars":     "#22C55E",
            "Casual Buyers":      "#EAB308",
            "At-Risk / Churning": "#EF4444"
        }
        descs = {
            "VIP Segment":        "High recency · High frequency · High spend",
            "Loyal Regulars":     "Consistent buyers with strong engagement",
            "Casual Buyers":      "Occasional purchases with moderate value",
            "At-Risk / Churning": "Declining activity — needs re-engagement"
        }
        col  = colors.get(pred_label, "#F97316")
        desc = descs.get(pred_label, "")

        st.markdown(f"""
        <div class="result-row">
          <div class="result-card">
            <span class="result-label">Customer Segment</span>
            <span class="result-value" style="color:{col};">{pred_label}</span>
            <span class="result-sub">{desc}</span>
          </div>
          <div class="result-card" style="max-width:320px;">
            <span class="result-label">AI Confidence</span>
            <span class="result-value" style="color:#F97316;">{conf:.1f}%</span>
            <span class="result-sub">Random Forest probability score</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("⚠️ Model files not found. Place customer_model.pkl, segment_encoder.pkl, and model_features.pkl in the same directory as app.py.")

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<p class="footer-text">
    Engineered by Shariq Adnan &nbsp;&middot;&nbsp; Boston Institute of Analytics Capstone 2026
</p>
""", unsafe_allow_html=True)