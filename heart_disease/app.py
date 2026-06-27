import streamlit as st
import pandas as pd
import joblib
import time

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor | Akarsh",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load Artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model   = joblib.load("KNN_heart.pkl")
    scaler  = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns

model, scaler, expected_columns = load_model()

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Import Google Font */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Main background */
  .main { background: linear-gradient(135deg, #0d1b2a 0%, #1a0a0a 100%); }
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, #1a0808 100%);
    border-right: 1px solid rgba(255,80,80,0.25);
  }

  /* Sidebar — all label text bright white */
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] .st-emotion-cache-1629p8f,
  [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.3px;
  }

  /* Sidebar — dropdown & number input VALUE text */
  [data-testid="stSidebar"] input,
  [data-testid="stSidebar"] .st-emotion-cache-1c7y2kl,
  [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] div {
    color: #ffd166 !important;
    font-weight: 600 !important;
  }

  /* Sidebar selectbox background */
  [data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,120,100,0.35) !important;
    border-radius: 8px !important;
    color: #ffd166 !important;
  }

  /* Sidebar number input box */
  [data-testid="stSidebar"] [data-baseweb="input"] input {
    background-color: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,120,100,0.35) !important;
    border-radius: 8px !important;
    color: #7efff5 !important;
    font-weight: 700 !important;
  }

  /* Sidebar slider value label */
  [data-testid="stSidebar"] [data-testid="stSlider"] p {
    color: #ff6b6b !important;
    font-weight: 700 !important;
  }

  /* Sidebar heading text */
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 {
    color: #ff6b6b !important;
    font-weight: 700 !important;
  }

  /* Sidebar markdown text */
  [data-testid="stSidebar"] .stMarkdown p {
    color: #aaaaaa !important;
    font-weight: 400 !important;
  }

  /* Title */
  .hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #ff4d4d, #ff9999);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.2rem;
  }
  .hero-sub {
    text-align: center;
    color: #9e9e9e;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
  }

  /* Metric cards */
  .metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,77,77,0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    backdrop-filter: blur(10px);
  }
  .metric-card h3 { color: #ff6b6b; font-size: 1.6rem; margin: 0; }
  .metric-card p  { color: #9e9e9e; font-size: 0.8rem; margin: 0; }

  /* Predict button */
  div.stButton > button {
    background: linear-gradient(135deg, #c0392b, #e74c3c);
    color: white;
    font-weight: 600;
    font-size: 1.1rem;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 2.5rem;
    cursor: pointer;
    width: 100%;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(231,76,60,0.35);
  }
  div.stButton > button:hover {
    background: linear-gradient(135deg, #a93226, #cb4335);
    box-shadow: 0 6px 25px rgba(231,76,60,0.55);
    transform: translateY(-1px);
  }

  /* Result boxes */
  .result-high {
    background: linear-gradient(135deg, rgba(192,57,43,0.30), rgba(231,76,60,0.20));
    border: 2px solid #e74c3c;
    border-radius: 16px;
    padding: 2.2rem 2rem;
    text-align: center;
    animation: pulse 1.5s infinite;
  }
  .result-low {
    background: linear-gradient(135deg, rgba(39,174,96,0.25), rgba(46,204,113,0.15));
    border: 2px solid #2ecc71;
    border-radius: 16px;
    padding: 2.2rem 2rem;
    text-align: center;
  }
  .result-high h2 { color: #ff4444; font-size: 2.4rem; font-weight: 800; margin: 0.4rem 0; }
  .result-low  h2 { color: #27ae60; font-size: 2.4rem; font-weight: 800; margin: 0.4rem 0; }
  .result-high p { color: #ffffff; font-size: 1.2rem; font-weight: 500; margin: 0.4rem 0; }
  .result-low  p { color: #ffffff; font-size: 1.2rem; font-weight: 500; margin: 0.4rem 0; }

  @keyframes pulse {
    0%   { box-shadow: 0 0 0 0   rgba(231,76,60,0.5); }
    70%  { box-shadow: 0 0 0 12px rgba(231,76,60,0.0); }
    100% { box-shadow: 0 0 0 0   rgba(231,76,60,0.0); }
  }

  /* Divider */
  hr { border-color: rgba(255,77,77,0.2); }

  /* Info box */
  .info-box {
    background: rgba(255,255,255,0.04);
    border-left: 3px solid #e74c3c;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    color: #b0b0b0;
    font-size: 0.85rem;
    margin-top: 1rem;
  }
</style>
""", unsafe_allow_html=True)

# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">❤️ Heart Disease Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Powered by K-Nearest Neighbors · Clinical Data Analysis · by <b>Akarsh</b></div>', unsafe_allow_html=True)
st.markdown("---")

# ── Sidebar ── Input Controls ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 Patient Details")
    st.markdown("Fill in the patient's clinical information below.")
    st.markdown("---")

    age      = st.slider("🔢 Age", 18, 100, 40, help="Patient age in years")
    sex      = st.selectbox("⚧ Sex", ["MALE", "FEMALE"])
    chest_pain = st.selectbox("💢 Chest Pain Type", ["ATA", "NAP", "TA", "ASY"],
                              help="ATA=Atypical Angina, NAP=Non-Anginal Pain, TA=Typical Angina, ASY=Asymptomatic")
    resting_bp = st.number_input("🩸 Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("🧪 Cholesterol (mg/dL)", 100, 600, 200)
    fasting_bs  = st.selectbox("🍬 Fasting Blood Sugar > 120 mg/dL", [0, 1],
                                format_func=lambda x: "Yes (>120)" if x == 1 else "No (≤120)")
    resting_ecg = st.selectbox("📈 Resting ECG", ["Normal", "ST", "LVH"],
                                help="Normal / ST-T wave abnormality / Left ventricular hypertrophy")
    max_hr      = st.slider("💓 Max Heart Rate Achieved", 60, 220, 150)
    ex_angina   = st.selectbox("🏃 Exercise-Induced Angina", ["Y", "N"])
    oldpeak     = st.slider("📉 Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1)
    st_slope    = st.selectbox("📊 ST Slope", ["Up", "Flat", "Down"])

    st.markdown("---")
    predict_btn = st.button("🔬 Predict Now")

# ── Main Panel ── Metric Summary ──────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><h3>{age}</h3><p>Age (years)</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><h3>{resting_bp}</h3><p>Blood Pressure (mmHg)</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><h3>{cholesterol}</h3><p>Cholesterol (mg/dL)</p></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card"><h3>{max_hr}</h3><p>Max Heart Rate</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Input Summary Table ───────────────────────────────────────────────────────
with st.expander("📋 View Full Input Summary", expanded=False):
    summary = {
        "Feature": ["Age", "Sex", "Chest Pain", "Resting BP", "Cholesterol",
                    "Fasting BS", "Resting ECG", "Max HR", "Exercise Angina", "Oldpeak", "ST Slope"],
        "Value": [str(age), sex, chest_pain, str(resting_bp), str(cholesterol),
                  str(fasting_bs), resting_ecg, str(max_hr), ex_angina, str(oldpeak), st_slope]
    }
    st.dataframe(pd.DataFrame(summary), width='stretch', hide_index=True)

st.markdown("---")

# ── Prediction Logic ──────────────────────────────────────────────────────────
result_placeholder = st.empty()

if predict_btn:
    with st.spinner("🔄 Analyzing patient data..."):
        time.sleep(0.8)

    raw_input = {
        'Age': age, 'RestingBP': resting_bp, 'Cholesterol': cholesterol,
        'FastingBS': fasting_bs, 'MaxHR': max_hr, 'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + ex_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df    = input_df[expected_columns]
    scaled      = scaler.transform(input_df)
    prediction  = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0]

    risk_pct = int(probability[1] * 100)

    if prediction == 1:
        result_placeholder.markdown(f"""
        <div class="result-high">
          <div style="font-size:3rem;">🚨</div>
          <h2>HIGH RISK of Heart Disease</h2>
          <p>Estimated Risk Probability: <b style="color:#ff6b6b;">{risk_pct}%</b></p>
          <p>Please consult a cardiologist immediately for further evaluation.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        result_placeholder.markdown(f"""
        <div class="result-low">
          <div style="font-size:3rem;">✅</div>
          <h2>LOW RISK of Heart Disease</h2>
          <p>Estimated Risk Probability: <b style="color:#2ecc71;">{risk_pct}%</b></p>
          <p>Maintain a healthy lifestyle and schedule regular check-ups.</p>
        </div>
        """, unsafe_allow_html=True)

    # Risk breakdown
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("🟢 Low Risk Probability",  f"{int(probability[0]*100)}%")
    with c2:
        st.metric("🔴 High Risk Probability", f"{int(probability[1]*100)}%")

else:
    result_placeholder.markdown("""
    <div style="text-align:center; color:#555; padding:2rem 0;">
      <div style="font-size:3.5rem;">🩺</div>
      <p style="font-size:1.1rem;">Fill in the patient details in the sidebar and click <b>Predict Now</b></p>
    </div>
    """, unsafe_allow_html=True)

# ── Info Footer ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="info-box">
  ⚠️ <b>Disclaimer:</b> This tool is for educational and research purposes only.
  It is NOT a substitute for professional medical advice, diagnosis, or treatment.
  Always consult a qualified healthcare provider for clinical decisions.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#444; font-size:0.8rem;'>Built with ❤️ using Streamlit · KNN Model · Scikit-Learn | © Akarsh</p>",
    unsafe_allow_html=True
)