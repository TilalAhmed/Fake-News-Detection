import streamlit as st
import joblib
import pickle
import numpy as np
import os
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

/* Hide default streamlit elements */
#MainMenu, footer, header {visibility: hidden;}

/* Top header */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3.2rem;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #ff6b35 0%, #f7c59f 50%, #efefd0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: #555570;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* Divider */
.styled-divider {
    height: 1px;
    background: linear-gradient(90deg, #ff6b35, transparent);
    margin: 1.5rem 0;
}

/* Textarea */
.stTextArea textarea {
    background: #13131f !important;
    border: 1px solid #2a2a3f !important;
    border-radius: 12px !important;
    color: #e8e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 1rem !important;
    transition: border-color 0.3s !important;
}
.stTextArea textarea:focus {
    border-color: #ff6b35 !important;
    box-shadow: 0 0 0 2px rgba(255,107,53,0.15) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #ff6b35, #e8520a) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    transition: opacity 0.2s, transform 0.15s !important;
    letter-spacing: 0.5px !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

/* Result cards */
.result-card {
    border-radius: 16px;
    padding: 1.8rem;
    margin-top: 1.2rem;
    position: relative;
    overflow: hidden;
}
.result-real {
    background: linear-gradient(135deg, #0d2b1f, #0a1f16);
    border: 1px solid #1a5c38;
}
.result-fake {
    background: linear-gradient(135deg, #2b0d0d, #1f0a0a);
    border: 1px solid #5c1a1a;
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    margin-bottom: 0.4rem;
}
.result-label-real { color: #4ade80; }
.result-label-fake { color: #f87171; }

.result-confidence {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

/* Model selector */
.stSelectbox > div > div {
    background: #13131f !important;
    border: 1px solid #2a2a3f !important;
    border-radius: 8px !important;
    color: #e8e8f0 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d0d18 !important;
    border-right: 1px solid #1a1a2e !important;
}
[data-testid="stSidebar"] .stMarkdown {
    color: #a0a0b8 !important;
}

/* Model badge */
.model-badge {
    display: inline-block;
    background: #1a1a2e;
    border: 1px solid #2a2a45;
    border-radius: 6px;
    padding: 0.3rem 0.8rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #ff6b35;
    margin-bottom: 1rem;
}

/* Stats row */
.stat-box {
    background: #13131f;
    border: 1px solid #2a2a3f;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.stat-number {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.6rem;
    color: #ff6b35;
}
.stat-label {
    font-size: 0.75rem;
    color: #555570;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'DM Mono', monospace;
}

/* Progress bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #ff6b35, #f7c59f) !important;
}

/* Info box */
.info-box {
    background: #13131f;
    border-left: 3px solid #ff6b35;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: #888;
    margin-top: 0.8rem;
    font-family: 'DM Mono', monospace;
}
</style>
""", unsafe_allow_html=True)


# ── Load models ───────────────────────────────────────────────────────────────
MODEL_DIR = "models"

MODEL_INFO = {
    "SVM": {
        "file": "SVC.pkl",
        "accuracy": "99.4%",
        "desc": "Support Vector Machine — highest accuracy, great for high-dimensional text"
    },
    "Logistic Regression": {
        "file": "LR.pkl",
        "accuracy": "98.6%",
        "desc": "Fast, interpretable baseline with strong performance"
    },
    "Decision Tree": {
        "file": "DT.pkl",
        "accuracy": "99.5%",
        "desc": "Rule-based; easy to explain in interviews"
    },
    "Naïve Bayes": {
        "file": "NB.pkl",
        "accuracy": "93.4%",
        "desc": "Probabilistic; extremely fast inference"
    },
    "Gradient Boosting": {
        "file": "GB.pkl",
        "accuracy": "99.5%",
        "desc": "Sequential ensemble — strong on tabular/text patterns"
    },
}

VECTORIZER_FILE = os.path.join(MODEL_DIR, "vectorizer.pkl")

@st.cache_resource
def load_assets():
    """Load vectorizer and all available models."""
    assets = {"vectorizer": None, "models": {}}

    if os.path.exists(VECTORIZER_FILE):
        with open(VECTORIZER_FILE, "rb") as f:
            assets["vectorizer"] = pickle.load(f)

    for name, info in MODEL_INFO.items():
        path = os.path.join(MODEL_DIR, info["file"])
        if os.path.exists(path):
            with open(path, "rb") as f:
                assets["models"][name] = pickle.load(f)

    return assets

assets = load_assets()
available_models = list(assets["models"].keys()) if assets["models"] else list(MODEL_INFO.keys())
models_loaded = len(assets["models"]) > 0


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:Syne,sans-serif; font-weight:800; font-size:1.1rem;
         color:#ff6b35; letter-spacing:1px; margin-bottom:0.3rem;'>
        FAKE NEWS DETECTOR
    </div>
    <div style='font-family:DM Mono,monospace; font-size:0.7rem; color:#444460;
         text-transform:uppercase; letter-spacing:2px; margin-bottom:1.5rem;'>
        v1.0 · Tilal Ahmed
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Select Model**")
    selected_model_name = st.selectbox(
        "Model",
        available_models,
        label_visibility="collapsed"
    )

    info = MODEL_INFO[selected_model_name]
    st.markdown(f"""
    <div class='info-box'>
        <b style='color:#ff6b35;'>{selected_model_name}</b><br>
        Accuracy: {info['accuracy']}<br><br>
        {info['desc']}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='styled-divider'></div>", unsafe_allow_html=True)

    st.markdown("**All Models**")
    for name, minfo in MODEL_INFO.items():
        dot = "🟢" if name in assets["models"] else "⚪"
        st.markdown(
            f"<div style='font-family:DM Mono,monospace; font-size:0.75rem; "
            f"color:#666; padding:0.2rem 0;'>{dot} {name} · {minfo['accuracy']}</div>",
            unsafe_allow_html=True
        )

    st.markdown("<div class='styled-divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:DM Mono,monospace; font-size:0.7rem; color:#333350;'>
        Built with Python · Scikit-learn<br>
        TF-IDF Vectorization<br>
        Trained on WELFake Dataset
    </div>
    """, unsafe_allow_html=True)


# ── Main content ──────────────────────────────────────────────────────────────
st.markdown("<div class='hero-title'>Fake News<br>Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>NLP · Machine Learning · Text Classification</div>", unsafe_allow_html=True)
st.markdown("<div class='styled-divider'></div>", unsafe_allow_html=True)

# Stats row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("<div class='stat-box'><div class='stat-number'>5</div><div class='stat-label'>Models</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='stat-box'><div class='stat-number'>99.5%</div><div class='stat-label'>Best Accuracy</div></div>", unsafe_allow_html=True)
with col3:
    st.markdown("<div class='stat-box'><div class='stat-number'>TF-IDF</div><div class='stat-label'>Vectorizer</div></div>", unsafe_allow_html=True)
with col4:
    st.markdown("<div class='stat-box'><div class='stat-number'>72K</div><div class='stat-label'>Training Samples</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Input + output layout
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown(f"<div class='model-badge'>▸ {selected_model_name}</div>", unsafe_allow_html=True)

    news_text = st.text_area(
        "Paste news article or headline",
        placeholder="Paste the news article, headline, or any text you want to verify...",
        height=220,
        label_visibility="collapsed"
    )

    char_count = len(news_text.strip())
    word_count = len(news_text.split()) if news_text.strip() else 0
    st.markdown(
        f"<div style='font-family:DM Mono,monospace; font-size:0.72rem; color:#333350; "
        f"text-align:right; margin-top:-0.5rem;'>{word_count} words · {char_count} chars</div>",
        unsafe_allow_html=True
    )

    analyze_btn = st.button("Analyze Article →")

with right:
    st.markdown("""
    <div style='background:#0d0d18; border:1px solid #1a1a2e; border-radius:12px;
         padding:1.2rem; margin-bottom:1rem;'>
        <div style='font-family:Syne,sans-serif; font-weight:700; color:#ff6b35;
             font-size:0.85rem; text-transform:uppercase; letter-spacing:1px;
             margin-bottom:0.8rem;'>How it works</div>
        <div style='font-family:DM Sans,sans-serif; font-size:0.82rem; color:#555570;
             line-height:1.7;'>
            1. Text is cleaned &amp; preprocessed<br>
            2. TF-IDF converts text to features<br>
            3. Model predicts Real or Fake<br>
            4. Confidence score is returned
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#0d0d18; border:1px solid #1a1a2e; border-radius:12px;
         padding:1.2rem;'>
        <div style='font-family:Syne,sans-serif; font-weight:700; color:#ff6b35;
             font-size:0.85rem; text-transform:uppercase; letter-spacing:1px;
             margin-bottom:0.8rem;'>Tips</div>
        <div style='font-family:DM Sans,sans-serif; font-size:0.82rem; color:#555570;
             line-height:1.7;'>
            • Paste full articles for best results<br>
            • Short headlines may be less accurate<br>
            • Try the same text across models<br>
            • SVM gives highest accuracy
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Prediction logic ──────────────────────────────────────────────────────────
if analyze_btn:
    if not news_text.strip():
        st.warning("Please paste some text to analyze.")
    elif word_count < 5:
        st.warning("Text is too short. Please paste at least a sentence or headline.")
    else:
        with st.spinner("Analyzing..."):
            time.sleep(0.6)  # slight pause for UX

            if models_loaded and assets["vectorizer"]:
                model = assets["models"][selected_model_name]
                vec = assets["vectorizer"].transform([news_text])
                pred = model.predict(vec)[0]
                label = "REAL" if pred == 1 else "FAKE"

                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(vec)[0]
                    confidence = max(proba) * 100
                elif hasattr(model, "decision_function"):
                    score = model.decision_function(vec)[0]
                    confidence = min(99.9, 50 + abs(score) * 15)
                else:
                    confidence = 85.0
            else:
                # Demo mode — no models loaded yet
                label = "FAKE" if any(w in news_text.lower() for w in
                                      ["aliens", "miracle", "secret", "shocking", "exposed"]) else "REAL"
                confidence = np.random.uniform(82, 97)

        # Display result
        is_real = label == "REAL"
        card_class = "result-real" if is_real else "result-fake"
        label_class = "result-label-real" if is_real else "result-label-fake"
        emoji = "✓" if is_real else "✗"
        verdict_text = "This article appears to be legitimate." if is_real else "This article shows signs of misinformation."

        st.markdown(f"""
        <div class='result-card {card_class}'>
            <div class='result-label {label_class}'>{emoji} {label}</div>
            <div style='font-family:DM Sans,sans-serif; font-size:0.9rem;
                 color:#888; margin-bottom:1rem;'>{verdict_text}</div>
            <div class='result-confidence'>Confidence: {confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(int(confidence))
        st.markdown(
            f"<div style='font-family:DM Mono,monospace; font-size:0.75rem; "
            f"color:#333350; text-align:center;'>Model: {selected_model_name} · "
            f"Confidence: {confidence:.1f}%</div>",
            unsafe_allow_html=True
        )

        if not models_loaded:
            st.info("⚠ Running in demo mode — place your trained .pkl files in the `models/` folder to use real predictions.")
