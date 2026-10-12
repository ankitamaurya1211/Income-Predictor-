"""
app.py
------
Streamlit web application for the Adult Income (Salary Prediction) model.
Run with:  streamlit run app.py
"""

import pathlib
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

# ─────────────────────────────────────────────
# Page config  (MUST be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Income Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Global CSS — professional dark-accent theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Root palette ── */
:root {
    --primary:   #2563eb;
    --primary-light: #dbeafe;
    --success:   #16a34a;
    --success-bg:#dcfce7;
    --warning:   #d97706;
    --warning-bg:#fef3c7;
    --surface:   #f8fafc;
    --border:    #e2e8f0;
    --text:      #0f172a;
    --muted:     #64748b;
    --radius:    12px;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }

/* ── App background ── */
.stApp { background: #f1f5f9; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #1e3a5f 0%, #0f172a 100%);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] hr { border-color: #334155 !important; }
[data-testid="stSidebar"] .sidebar-metric-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
[data-testid="stSidebar"] .sidebar-metric-card .val {
    font-size: 1.5rem;
    font-weight: 700;
    color: #93c5fd !important;
}
[data-testid="stSidebar"] .sidebar-metric-card .lbl {
    font-size: 0.75rem;
    color: #94a3b8 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
    border-radius: var(--radius);
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
}
.hero-banner h1 {
    font-size: 2rem;
    font-weight: 800;
    margin: 0 0 0.4rem 0;
    color: white !important;
}
.hero-banner p {
    font-size: 1rem;
    color: #bfdbfe;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: #e0f2fe;
    margin-top: 0.75rem;
}

/* ── Section card ── */
.section-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.section-title {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── Form labels & inputs ── */
label { font-weight: 500 !important; color: var(--text) !important; }
.stNumberInput input, .stSelectbox select {
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
}
.stNumberInput input:focus, .stSelectbox select:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
}

/* ── Submit button ── */
.stFormSubmitButton button {
    background: linear-gradient(135deg, #1e3a5f, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 0.65rem 2rem !important;
    letter-spacing: 0.02em;
    transition: opacity 0.2s;
}
.stFormSubmitButton button:hover { opacity: 0.88 !important; }

/* ── Result cards ── */
.result-card {
    border-radius: var(--radius);
    padding: 1.5rem 2rem;
    margin-bottom: 1rem;
    border-left: 6px solid;
}
.result-card.high {
    background: var(--success-bg);
    border-color: var(--success);
}
.result-card.low {
    background: var(--warning-bg);
    border-color: var(--warning);
}
.result-card .rc-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
}
.result-card .rc-value {
    font-size: 1.9rem;
    font-weight: 800;
    margin: 4px 0;
}
.result-card.high .rc-value { color: var(--success); }
.result-card.low  .rc-value { color: var(--warning); }
.result-card .rc-sub {
    font-size: 0.88rem;
    color: var(--muted);
}

/* ── Confidence bar ── */
.conf-bar-wrap {
    background: var(--border);
    border-radius: 999px;
    height: 10px;
    width: 100%;
    overflow: hidden;
    margin-top: 6px;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
}

/* ── Prob grid ── */
.prob-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-top: 0.5rem;
}
.prob-cell {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
}
.prob-cell .pc-label {
    font-size: 0.75rem;
    color: var(--muted);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.prob-cell .pc-value {
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--text);
    margin: 4px 0 6px 0;
}

/* ── Input summary table ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Footer ── */
.app-footer {
    text-align: center;
    color: var(--muted);
    font-size: 0.76rem;
    padding: 1.5rem 0 0.5rem 0;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load model artifact (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_artifact():
    if not MODEL_PATH.exists():
        st.error("Model file not found. Please run `python train_model.py` first.")
        st.stop()
    return joblib.load(MODEL_PATH)


artifact             = load_artifact()
pipeline             = artifact["pipeline"]
model_name           = artifact["model_name"]
feature_columns      = artifact["feature_columns"]
numeric_features     = artifact["numeric_features"]
categorical_features = artifact["categorical_features"]
target_classes       = artifact["target_classes"]
category_options     = artifact["category_options"]

# ─────────────────────────────────────────────
# Sidebar — model info & quick stats
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💼 Income Predictor")
    st.markdown("---")

    st.markdown("**About this App**")
    st.markdown(
        "Uses the **UCI Adult Income** dataset to predict whether "
        "an individual earns **≤ $50K** or **> $50K** per year."
    )
    st.markdown("---")

    # Model stats cards
    st.markdown("**Model Info**")
    for lbl, val in [
        ("Algorithm",   model_name),
        ("Features",    str(len(feature_columns))),
        ("Train rows",  "36,177"),
        ("CV Metric",   "ROC-AUC"),
        ("AUC Score",   "0.9239"),
        ("Accuracy",    "86.5 %"),
    ]:
        st.markdown(
            f'<div class="sidebar-metric-card">'
            f'<div class="lbl">{lbl}</div>'
            f'<div class="val">{val}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        '<span style="font-size:0.72rem;color:#64748b;">'
        "Source: UCI ML Repository · Adult Income Dataset"
        "</span>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# Hero banner
# ─────────────────────────────────────────────
st.markdown(
    f"""
    <div class="hero-banner">
        <h1>💼 Income Prediction</h1>
        <p>Enter an individual's census information below to predict their annual income bracket.</p>
        <span class="hero-badge">🤖 Model: {model_name}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Input form
# ─────────────────────────────────────────────
st.markdown(
    '<div class="section-title">📋 APPLICANT DETAILS</div>',
    unsafe_allow_html=True,
)

with st.form("prediction_form"):
    # ── Row 1: Demographics ───────────────────
    st.markdown(
        '<div class="section-title">👤 Demographics</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        age = st.number_input("Age", min_value=17, max_value=90, value=35, step=1)
    with c2:
        gender = st.selectbox("Gender", options=category_options.get("gender", []))
    with c3:
        race = st.selectbox("Race", options=category_options.get("race", []))
    with c4:
        native_country = st.selectbox(
            "Native Country", options=category_options.get("native-country", [])
        )

    st.divider()

    # ── Row 2: Education ──────────────────────
    st.markdown(
        '<div class="section-title">🎓 Education</div>',
        unsafe_allow_html=True,
    )
    c5, c6 = st.columns(2)
    with c5:
        education = st.selectbox(
            "Education Level", options=category_options.get("education", [])
        )
    with c6:
        educational_num = st.number_input(
            "Education (numeric)",
            min_value=1, max_value=16, value=10, step=1,
            help="1 = Preschool · 9 = HS-grad · 13 = Bachelors · 16 = Doctorate",
        )

    st.divider()

    # ── Row 3: Employment ─────────────────────
    st.markdown(
        '<div class="section-title">💼 Employment</div>',
        unsafe_allow_html=True,
    )
    c7, c8, c9, c10 = st.columns(4)
    with c7:
        workclass = st.selectbox(
            "Work Class", options=category_options.get("workclass", [])
        )
    with c8:
        occupation = st.selectbox(
            "Occupation", options=category_options.get("occupation", [])
        )
    with c9:
        marital_status = st.selectbox(
            "Marital Status", options=category_options.get("marital-status", [])
        )
    with c10:
        relationship = st.selectbox(
            "Relationship", options=category_options.get("relationship", [])
        )

    c11, c12 = st.columns(2)
    with c11:
        hours_per_week = st.number_input(
            "Hours per Week", min_value=1, max_value=99, value=40, step=1
        )

    st.divider()

    # ── Row 4: Financials ─────────────────────
    st.markdown(
        '<div class="section-title">💰 Financial</div>',
        unsafe_allow_html=True,
    )
    c13, c14 = st.columns(2)
    with c13:
        capital_gain = st.number_input(
            "Capital Gain ($)", min_value=0, max_value=100000, value=0, step=100
        )
    with c14:
        capital_loss = st.number_input(
            "Capital Loss ($)", min_value=0, max_value=4356, value=0, step=50
        )

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button(
        "🔍  Predict Income Bracket", use_container_width=True
    )

# ─────────────────────────────────────────────
# Prediction & results
# ─────────────────────────────────────────────
if submitted:
    input_data = {
        "age":             age,
        "workclass":       workclass,
        "education":       education,
        "educational-num": educational_num,
        "marital-status":  marital_status,
        "occupation":      occupation,
        "relationship":    relationship,
        "race":            race,
        "gender":          gender,
        "capital-gain":    capital_gain,
        "capital-loss":    capital_loss,
        "hours-per-week":  hours_per_week,
        "native-country":  native_country,
    }

    input_df        = pd.DataFrame([input_data])[feature_columns]
    prediction_idx  = pipeline.predict(input_df)[0]
    probabilities   = pipeline.predict_proba(input_df)[0]
    predicted_label = target_classes[prediction_idx]
    confidence      = float(probabilities[prediction_idx]) * 100
    prob_high       = float(probabilities[1]) * 100   # >50K probability
    prob_low        = float(probabilities[0]) * 100   # <=50K probability

    st.markdown("---")
    st.markdown(
        '<div class="section-title">📊 PREDICTION RESULT</div>',
        unsafe_allow_html=True,
    )

    res_col, conf_col = st.columns([1.4, 1])

    # ── Result card ───────────────────────────
    with res_col:
        if predicted_label == ">50K":
            card_cls  = "high"
            icon      = "🟢"
            label_txt = "HIGH INCOME"
            sub_txt   = "Predicted annual income exceeds $50,000"
        else:
            card_cls  = "low"
            icon      = "🟡"
            label_txt = "MODERATE INCOME"
            sub_txt   = "Predicted annual income is $50,000 or below"

        st.markdown(
            f"""
            <div class="result-card {card_cls}">
                <div class="rc-label">{icon} &nbsp; Prediction</div>
                <div class="rc-value">{predicted_label}</div>
                <div class="rc-sub">{sub_txt}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Probability breakdown grid
        bar_high_color = "#16a34a"
        bar_low_color  = "#d97706"
        st.markdown(
            f"""
            <div class="prob-grid">
                <div class="prob-cell">
                    <div class="pc-label">Prob ≤ $50K</div>
                    <div class="pc-value">{prob_low:.1f}%</div>
                    <div class="conf-bar-wrap">
                        <div class="conf-bar-fill"
                             style="width:{prob_low:.1f}%;background:{bar_low_color};">
                        </div>
                    </div>
                </div>
                <div class="prob-cell">
                    <div class="pc-label">Prob > $50K</div>
                    <div class="pc-value">{prob_high:.1f}%</div>
                    <div class="conf-bar-wrap">
                        <div class="conf-bar-fill"
                             style="width:{prob_high:.1f}%;background:{bar_high_color};">
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Confidence gauge (native metric + progress) ───
    with conf_col:
        st.metric(
            label="Model Confidence",
            value=f"{confidence:.1f}%",
            delta="High" if confidence >= 75 else ("Medium" if confidence >= 55 else "Low"),
        )
        st.progress(int(confidence))

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Probability Chart**")
        prob_df = pd.DataFrame(
            {"Income Bracket": ["≤ $50K", "> $50K"],
             "Probability":    [round(prob_low / 100, 4), round(prob_high / 100, 4)]}
        ).set_index("Income Bracket")
        st.bar_chart(prob_df, color=["#2563eb"])

    # ── Input summary ──────────────────────────
    with st.expander("🔎 View Input Data Sent to Model"):
        styled = input_df.T.rename(columns={0: "Value"})
        st.dataframe(styled, use_container_width=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown(
    f"""
    <div class="app-footer">
        Adult Income Dataset &nbsp;·&nbsp; UCI Machine Learning Repository
        &nbsp;·&nbsp; Model: <strong>{model_name}</strong>
        &nbsp;·&nbsp; Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
