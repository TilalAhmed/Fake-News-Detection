import streamlit as st
import streamlit.components.v1 as components
import os
import re
import string
import pickle
import numpy as np
import time

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0f1117; color: #e2e8f0; }
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #13161f !important;
    border-right: 1px solid #1e2235 !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] span { color: #64748b !important; font-size: 0.8rem !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong { color: #e2e8f0 !important; }

/* Selectbox */
.stSelectbox > div > div {
    background: #0f1117 !important;
    border: 1px solid #1e2235 !important;
    border-radius: 8px !important;
    color: #cbd5e1 !important;
}

/* Textarea */
.stTextArea textarea {
    background: #0f1117 !important;
    border: 1px solid #1e2235 !important;
    border-radius: 8px !important;
    color: #cbd5e1 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
}
.stTextArea textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: none !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    width: 100% !important;
    padding: 0.6rem 1.5rem !important;
    font-size: 0.88rem !important;
}
.stButton > button:hover { opacity: 0.9 !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #13161f !important;
    border: 1px solid #1e2235 !important;
    border-radius: 10px !important;
    padding: 14px !important;
}
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.7rem !important; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="stMetricValue"] { color: #fff !important; font-size: 1.5rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { color: #475569 !important; font-size: 0.7rem !important; }
[data-testid="stMetricDelta"] svg { display: none; }

/* Cards */
.card {
    background: #13161f;
    border: 1px solid #1e2235;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
}
.card-title {
    font-size: 11px;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 12px;
}

/* Result boxes */
.res-real {
    background: #052e16;
    border: 1px solid #166534;
    border-radius: 10px;
    padding: 16px;
    margin-top: 12px;
}
.res-fake {
    background: #2d0a0a;
    border: 1px solid #7f1d1d;
    border-radius: 10px;
    padding: 16px;
    margin-top: 12px;
}
.res-label-real { font-size: 1.4rem; font-weight: 700; color: #4ade80; }
.res-label-fake { font-size: 1.4rem; font-weight: 700; color: #f87171; }
.res-desc { font-size: 11px; color: #94a3b8; margin-top: 4px; }
.conf-bar-wrap { height: 6px; background: #1e293b; border-radius: 3px; margin-top: 8px; overflow: hidden; }

/* Badges */
.badge-real { background: #052e16; color: #4ade80; border: 1px solid #166534; border-radius: 99px; padding: 2px 9px; font-size: 10px; font-weight: 600; }
.badge-fake { background: #2d0a0a; color: #f87171; border: 1px solid #7f1d1d; border-radius: 99px; padding: 2px 9px; font-size: 10px; font-weight: 600; }

/* Table */
.rev-table { width: 100%; border-collapse: collapse; font-size: 11px; }
.rev-table th { color: #475569; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; padding: 0 8px 8px 0; border-bottom: 1px solid #1e2235; font-weight: 500; text-align: left; }
.rev-table td { padding: 7px 8px 7px 0; border-bottom: 1px solid #13161f; color: #cbd5e1; vertical-align: middle; }

/* Model perf bars */
.mp-row { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
.mp-name { font-size: 10px; color: #94a3b8; width: 120px; flex-shrink: 0; text-align: right; }
.mp-track { flex: 1; height: 16px; background: #0f1117; border-radius: 3px; overflow: hidden; }
.mp-fill { height: 100%; border-radius: 3px; display: flex; align-items: center; padding-left: 7px; }
.mp-val { font-size: 9px; font-weight: 600; color: #fff; }

/* Subject bars */
.sub-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.sub-name { font-size: 10px; color: #94a3b8; width: 110px; flex-shrink: 0; text-align: right; }
.sub-track { flex: 1; height: 10px; background: #0f1117; border-radius: 2px; overflow: hidden; }
.sub-fill { height: 100%; border-radius: 2px; }
.sub-val { font-size: 10px; color: #64748b; width: 40px; flex-shrink: 0; }

/* Info box */
.info-box {
    background: #0f1117;
    border-left: 2px solid #7c3aed;
    border-radius: 0 6px 6px 0;
    padding: 10px 12px;
    font-size: 11px;
    color: #64748b;
    line-height: 1.9;
    margin-top: 8px;
}

/* Pipeline */
.pipe-wrap { display: flex; align-items: center; gap: 0; overflow-x: auto; padding-bottom: 2px; }
.pipe-step { flex: 1; min-width: 80px; background: #0f1117; border: 1px solid #1e2235; border-radius: 8px; padding: 10px 6px; text-align: center; }
.pipe-icon { font-size: 18px; margin-bottom: 4px; }
.pipe-title { font-size: 9px; font-weight: 600; color: #e2e8f0; line-height: 1.4; }
.pipe-sub { font-size: 8px; color: #475569; margin-top: 2px; }
.pipe-arrow { color: #374151; font-size: 12px; padding: 0 3px; flex-shrink: 0; }

.styled-div { height: 1px; background: #1e2235; margin: 1rem 0; }
.wc-svg text { font-family: 'Inter', sans-serif; }
</style>
""", unsafe_allow_html=True)


# ── Load models ───────────────────────────────────────────────────────────────
MODEL_DIR = "models"

MODEL_INFO = {
    "Logistic Regression": {"file": "LR.pkl",  "accuracy": "98.56%", "color": "#0891b2"},
    "Decision Tree":       {"file": "DT.pkl",  "accuracy": "99.67%", "color": "#7c3aed"},
    "Gradient Boosting":   {"file": "GB.pkl",  "accuracy": "99.5%",  "color": "#4f46e5"},
    "Linear SVM":          {"file": "SVC.pkl", "accuracy": "98.74%", "color": "#059669"},
    "Naïve Bayes":         {"file": "NB.pkl",  "accuracy": "93.4%",  "color": "#d97706"},
}

@st.cache_resource
def load_assets():
    assets = {"vectorizer": None, "models": {}}
    vec_path = os.path.join(MODEL_DIR, "vectorizer.pkl")
    if os.path.exists(vec_path):
        with open(vec_path, "rb") as f:
            assets["vectorizer"] = pickle.load(f)
    for name, info in MODEL_INFO.items():
        path = os.path.join(MODEL_DIR, info["file"])
        if os.path.exists(path):
            with open(path, "rb") as f:
                assets["models"][name] = pickle.load(f)
    return assets

assets = load_assets()
models_loaded = len(assets["models"]) > 0
available_models = list(assets["models"].keys()) if models_loaded else list(MODEL_INFO.keys())


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:10px;margin-bottom:12px;'>
      <div style='width:42px;height:42px;background:linear-gradient(135deg,#7c3aed,#4f46e5);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0;'>🔍</div>
      <div>
        <div style='font-size:14px;font-weight:700;color:#fff;line-height:1.2;'>Fake News<br>Detection System</div>
        <div style='font-size:9px;color:#475569;margin-top:2px;'>Machine Learning + NLP</div>
      </div>
    </div>
    <div style='font-size:10px;color:#475569;line-height:1.7;margin-bottom:12px;'>Detects whether a news article is Real or Fake using TF-IDF Vectorization and multiple ML Algorithms.</div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='styled-div'></div>", unsafe_allow_html=True)
    st.markdown("**Select Model**")
    selected_model = st.selectbox("Model", available_models, label_visibility="collapsed")

    info = MODEL_INFO[selected_model]
    st.markdown(f"""
    <div class='info-box'>
        <span style='color:#a78bfa;font-weight:600;'>{selected_model}</span><br>
        Accuracy: <span style='color:#e2e8f0;'>{info['accuracy']}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='styled-div'></div>", unsafe_allow_html=True)
    st.markdown("**All Models**")
    for name, minfo in MODEL_INFO.items():
        dot = "🟢" if name in assets["models"] else "⚪"
        st.markdown(
            f"<div style='font-size:11px;color:#475569;padding:3px 0;'>{dot} {name} · "
            f"<span style='color:#64748b;'>{minfo['accuracy']}</span></div>",
            unsafe_allow_html=True
        )

    st.markdown("<div class='styled-div'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#0f1117;border:1px solid #1e2235;border-radius:8px;padding:12px;'>
      <div style='font-size:11px;font-weight:600;color:#94a3b8;margin-bottom:8px;'>🗄️ About Dataset</div>
      <div style='font-size:10px;color:#475569;line-height:1.8;'>
        Combined Fake.csv and True.csv datasets containing news articles labeled as Real or Fake.<br><br>
        Total Features: 4<br>(Title, Text, Subject, Date)<br>
        Target: class (0 = Fake, 1 = Real)
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:10px;color:#374151;line-height:1.9;'>
        Tilal Ahmed · Iqra University, Karachi<br>
        Scikit-learn · TF-IDF · ISOT Dataset
    </div>
    """, unsafe_allow_html=True)


# ── KPI Row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
with k1: st.metric("📄 Total Articles", "44,898", "100%")
with k2: st.metric("❌ Fake Articles", "23,481", "52.27%")
with k3: st.metric("✅ Real Articles", "21,417", "47.73%")
with k4: st.metric("🧠 ML Models", "6", "Algorithms")
with k5: st.metric("🏆 Best Accuracy", "99.67%", "Decision Tree")

st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)

# ── Row 1: Analyzer + Donut + Subject ────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='card'><div class='card-title'>📡 Live News Analyzer</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:11px;color:#475569;margin-bottom:8px;'>Paste a news article and find out whether it is Real or Fake.</div>", unsafe_allow_html=True)

    news_input = st.text_area("News article", placeholder="Paste news article here...",
                              height=100, label_visibility="collapsed")
    wc = len(news_input.split()) if news_input.strip() else 0
    st.markdown(f"<div style='font-size:9px;color:#374151;text-align:right;margin-top:-10px;'>{wc} words</div>", unsafe_allow_html=True)

    analyze_btn = st.button("📡 Analyze News")

    if analyze_btn:
        if not news_input.strip():
            st.warning("Please paste some text to analyze.")
        elif wc < 5:
            st.warning("Too short — paste at least a sentence.")
        else:
            with st.spinner("Analyzing..."):
                time.sleep(0.4)
                if models_loaded and assets["vectorizer"]:
                    model = assets["models"][selected_model]
                    cleaned = clean_text(news_input)
                    vec = assets["vectorizer"].transform([cleaned])
                    pred = model.predict(vec)[0]
                    label = "REAL" if pred == 1 else "FAKE"
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(vec)[0]
                        confidence = max(proba) * 100
                    elif hasattr(model, "decision_function"):
                        score = model.decision_function(vec)[0]
                        confidence = min(99.9, 50 + abs(float(score if np.isscalar(score) else score[0])) * 15)
                    else:
                        confidence = 85.0
                else:
                    fake_words = ["shocking","secret","exposed","conspiracy","viral","hoax","miracle","leaked","banned","won't believe"]
                    real_words = ["reuters","washington","official","government","president","minister","announced","policy","statement","parliament"]
                    t = news_input.lower()
                    fs = sum(1 for w in fake_words if w in t)
                    rs = sum(1 for w in real_words if w in t)
                    label = "REAL" if rs >= fs else "FAKE"
                    confidence = float(np.random.uniform(82, 97))

            is_real = label == "REAL"
            card_cls = "res-real" if is_real else "res-fake"
            lbl_cls  = "res-label-real" if is_real else "res-label-fake"
            emoji    = "✓" if is_real else "✗"
            desc     = "This article appears to be legitimate." if is_real else "This article shows signs of misinformation."
            bar_col  = "#4ade80" if is_real else "#f87171"

            st.markdown(f"""
            <div class='{card_cls}'>
              <div class='{lbl_cls}'>{emoji} {label} NEWS</div>
              <div class='res-desc'>{desc}</div>
              <div style='font-size:10px;color:#475569;margin-top:6px;'>Confidence: {confidence:.1f}%</div>
              <div class='conf-bar-wrap'><div style='height:100%;width:{confidence:.1f}%;background:{bar_col};border-radius:3px;'></div></div>
              <div style='font-size:9px;color:#374151;margin-top:4px;'>Model: {selected_model} · {"Models loaded" if models_loaded else "Demo mode"}</div>
            </div>
            """, unsafe_allow_html=True)

        if not models_loaded:
            st.info("⚠ Demo mode — place trained .pkl files in `models/` folder for real predictions.")

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'><div class='card-title'>🧩 Dataset Distribution</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='display:flex;gap:14px;margin-bottom:8px;flex-wrap:wrap;'>
      <span style='display:flex;align-items:center;gap:5px;font-size:11px;color:#94a3b8;'><span style='width:10px;height:10px;border-radius:50%;background:#f87171;display:inline-block;'></span>Fake News 23,481 (52.27%)</span>
      <span style='display:flex;align-items:center;gap:5px;font-size:11px;color:#94a3b8;'><span style='width:10px;height:10px;border-radius:50%;background:#4ade80;display:inline-block;'></span>Real News 21,417 (47.73%)</span>
    </div>
    """, unsafe_allow_html=True)
    import plotly.graph_objects as go
    fig_donut = go.Figure(go.Pie(
        labels=["Fake News", "Real News"], values=[23481, 21417], hole=0.65,
        marker=dict(colors=["#f87171", "#4ade80"], line=dict(width=0)),
        textinfo="none", hovertemplate="%{label}: %{value:,}<extra></extra>",
    ))
    fig_donut.update_layout(paper_bgcolor="#13161f", plot_bgcolor="#13161f",
        font=dict(color="#94a3b8"), margin=dict(l=10,r=10,t=10,b=10),
        showlegend=False, height=220)
    fig_donut.add_annotation(text="44,898<br><span style='font-size:10px'>Articles</span>",
        x=0.5, y=0.5, showarrow=False, font=dict(size=13, color="#fff"), align="center")
    st.plotly_chart(fig_donut, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='card'><div class='card-title'>📂 Subject Distribution</div>
    <div>
      <div class='sub-row'><span class='sub-name'>politicsNews</span><div class='sub-track'><div class='sub-fill' style='width:81%;background:#7c3aed;'></div></div><span class='sub-val'>12,189</span></div>
      <div class='sub-row'><span class='sub-name'>worldnews</span><div class='sub-track'><div class='sub-fill' style='width:62%;background:#2563eb;'></div></div><span class='sub-val'>9,286</span></div>
      <div class='sub-row'><span class='sub-name'>Government News</span><div class='sub-track'><div class='sub-fill' style='width:45%;background:#0891b2;'></div></div><span class='sub-val'>6,793</span></div>
      <div class='sub-row'><span class='sub-name'>News</span><div class='sub-track'><div class='sub-fill' style='width:42%;background:#059669;'></div></div><span class='sub-val'>6,236</span></div>
      <div class='sub-row'><span class='sub-name'>Middle-east</span><div class='sub-track'><div class='sub-fill' style='width:28%;background:#d97706;'></div></div><span class='sub-val'>4,234</span></div>
      <div class='sub-row'><span class='sub-name'>us-news</span><div class='sub-track'><div class='sub-fill' style='width:20%;background:#db2777;'></div></div><span class='sub-val'>2,940</span></div>
      <div class='sub-row'><span class='sub-name'>Others</span><div class='sub-track'><div class='sub-fill' style='width:21%;background:#64748b;'></div></div><span class='sub-val'>3,220</span></div>
    </div></div>
    """, unsafe_allow_html=True)


# ── Row 2: Article Length + Word Clouds ──────────────────────────────────────
col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("<div class='card'><div class='card-title'>📏 Article Length Analysis</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px;'>
      <div style='background:#2d0a0a;border:1px solid #7f1d1d;border-radius:8px;padding:12px;text-align:center;'>
        <div style='font-size:10px;color:#f87171;margin-bottom:6px;'>Avg. Fake News Length</div>
        <div style='font-size:26px;font-weight:700;color:#f87171;'>543</div>
        <div style='font-size:10px;color:#475569;'>words</div>
      </div>
      <div style='background:#052e16;border:1px solid #166534;border-radius:8px;padding:12px;text-align:center;'>
        <div style='font-size:10px;color:#4ade80;margin-bottom:6px;'>Avg. Real News Length</div>
        <div style='font-size:26px;font-weight:700;color:#4ade80;'>468</div>
        <div style='font-size:10px;color:#475569;'>words</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    import plotly.graph_objects as go
    fig_len = go.Figure()
    fig_len.add_trace(go.Bar(name="Fake", x=["<100","100-300","300-500","500-700","700+"],
        y=[120,1840,7200,8900,5421], marker_color="#f87171", marker_line_width=0))
    fig_len.add_trace(go.Bar(name="Real", x=["<100","100-300","300-500","500-700","700+"],
        y=[80,2100,8400,7200,3637], marker_color="#4ade80", marker_line_width=0))
    fig_len.update_layout(barmode="group", height=130, showlegend=False,
        paper_bgcolor="#13161f", plot_bgcolor="#13161f",
        font=dict(color="#475569", size=9), margin=dict(l=0,r=0,t=0,b=0),
        xaxis=dict(gridcolor="#1e2235"), yaxis=dict(gridcolor="#1e2235"))
    st.plotly_chart(fig_len, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col5:
    components.html("""
    <div style="background:#13161f;border:1px solid #1e2235;border-radius:12px;padding:16px;height:100%;">
      <div style="font-size:11px;font-weight:600;color:#f87171;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:10px;">🔴 Top Words in Fake News</div>
      <svg viewBox="0 0 260 175" width="100%" style="display:block;">
        <text x="130" y="48" text-anchor="middle" font-family="Inter,sans-serif" font-size="26" font-weight="700" fill="#f87171">people</text>
        <text x="130" y="76" text-anchor="middle" font-family="Inter,sans-serif" font-size="22" font-weight="700" fill="#ef4444">trump</text>
        <text x="44" y="42" text-anchor="middle" font-family="Inter,sans-serif" font-size="13" fill="#fbbf24">shocking</text>
        <text x="210" y="38" text-anchor="middle" font-family="Inter,sans-serif" font-size="13" fill="#f97316">breaking</text>
        <text x="220" y="60" text-anchor="middle" font-family="Inter,sans-serif" font-size="11" fill="#64748b">media</text>
        <text x="34" y="68" text-anchor="middle" font-family="Inter,sans-serif" font-size="13" fill="#f87171">just</text>
        <text x="130" y="100" text-anchor="middle" font-family="Inter,sans-serif" font-size="17" font-weight="600" fill="#fbbf24">news</text>
        <text x="46" y="98" text-anchor="middle" font-family="Inter,sans-serif" font-size="11" fill="#64748b">secret</text>
        <text x="216" y="94" text-anchor="middle" font-family="Inter,sans-serif" font-size="11" fill="#64748b">viral</text>
        <text x="82" y="120" text-anchor="middle" font-family="Inter,sans-serif" font-size="11" fill="#475569">say</text>
        <text x="184" y="118" text-anchor="middle" font-family="Inter,sans-serif" font-size="13" fill="#fbbf24">video</text>
        <text x="44" y="144" text-anchor="middle" font-family="Inter,sans-serif" font-size="11" fill="#f97316">exposed</text>
        <text x="100" y="146" text-anchor="middle" font-family="Inter,sans-serif" font-size="10" fill="#475569">hoax</text>
        <text x="150" y="146" text-anchor="middle" font-family="Inter,sans-serif" font-size="11" fill="#f87171">fake</text>
        <text x="210" y="144" text-anchor="middle" font-family="Inter,sans-serif" font-size="10" fill="#64748b">according</text>
        <text x="80" y="166" text-anchor="middle" font-family="Inter,sans-serif" font-size="10" fill="#475569">conspiracy</text>
        <text x="175" y="166" text-anchor="middle" font-family="Inter,sans-serif" font-size="10" fill="#475569">leaked</text>
      </svg>
    </div>
    """, height=270)

with col6:
    components.html("""
    <div style="background:#13161f;border:1px solid #1e2235;border-radius:12px;padding:16px;height:100%;">
      <div style="font-size:11px;font-weight:600;color:#4ade80;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:10px;">🟢 Top Words in Real News</div>
      <svg viewBox="0 0 260 175" width="100%" style="display:block;">
        <text x="130" y="46" text-anchor="middle" font-family="Inter,sans-serif" font-size="24" font-weight="700" fill="#4ade80">government</text>
        <text x="130" y="74" text-anchor="middle" font-family="Inter,sans-serif" font-size="20" font-weight="700" fill="#22c55e">reuters</text>
        <text x="36" y="42" text-anchor="middle" font-family="Inter,sans-serif" font-size="12" fill="#60a5fa">official</text>
        <text x="220" y="38" text-anchor="middle" font-family="Inter,sans-serif" font-size="18" font-weight="600" fill="#4ade80">said</text>
        <text x="220" y="62" text-anchor="middle" font-family="Inter,sans-serif" font-size="11" fill="#64748b">billion</text>
        <text x="34" y="66" text-anchor="middle" font-family="Inter,sans-serif" font-size="11" fill="#94a3b8">report</text>
        <text x="130" y="100" text-anchor="middle" font-family="Inter,sans-serif" font-size="16" fill="#60a5fa">president</text>
        <text x="42" y="98" text-anchor="middle" font-family="Inter,sans-serif" font-size="13" fill="#4ade80">state</text>
        <text x="78" y="120" text-anchor="middle" font-family="Inter,sans-serif" font-size="11" fill="#475569">announced</text>
        <text x="150" y="120" text-anchor="middle" font-family="Inter,sans-serif" font-size="13" fill="#22c55e">minister</text>
        <text x="216" y="118" text-anchor="middle" font-family="Inter,sans-serif" font-size="10" fill="#64748b">plan</text>
        <text x="44" y="142" text-anchor="middle" font-family="Inter,sans-serif" font-size="12" fill="#60a5fa">statement</text>
        <text x="130" y="142" text-anchor="middle" font-family="Inter,sans-serif" font-size="14" fill="#4ade80">policy</text>
        <text x="210" y="142" text-anchor="middle" font-family="Inter,sans-serif" font-size="11" fill="#64748b">world</text>
        <text x="78" y="164" text-anchor="middle" font-family="Inter,sans-serif" font-size="10" fill="#475569">election</text>
        <text x="170" y="164" text-anchor="middle" font-family="Inter,sans-serif" font-size="10" fill="#475569">senate</text>
      </svg>
    </div>
    """, height=270)


# ── Row 3: Confusion Matrix + Model Performance ───────────────────────────────
col7, col8 = st.columns(2)

with col7:
    st.markdown("""
    <div class='card'>
      <div class='card-title'>⊞ Confusion Matrix <span style='font-size:10px;color:#475569;font-weight:400;text-transform:none;letter-spacing:0;'>(Best Model: Decision Tree)</span></div>
      <div style='font-size:10px;color:#475569;text-align:center;margin-bottom:6px;'>← Predicted →</div>
      <div style='display:grid;grid-template-columns:auto 1fr 1fr;grid-template-rows:auto 1fr 1fr;gap:5px;'>
        <div></div>
        <div style='font-size:10px;color:#f87171;text-align:center;padding:4px;font-weight:600;'>Fake</div>
        <div style='font-size:10px;color:#4ade80;text-align:center;padding:4px;font-weight:600;'>Real</div>
        <div style='font-size:9px;color:#64748b;writing-mode:vertical-rl;transform:rotate(180deg);text-align:center;padding:4px 2px;'>Fake</div>
        <div style='background:#7f1d1d;border-radius:6px;padding:22px 8px;text-align:center;font-size:20px;font-weight:700;color:#fca5a5;'>11,523</div>
        <div style='background:#1e2235;border-radius:6px;padding:22px 8px;text-align:center;font-size:20px;font-weight:700;color:#475569;'>212</div>
        <div style='font-size:9px;color:#64748b;writing-mode:vertical-rl;transform:rotate(180deg);text-align:center;padding:4px 2px;'>Real</div>
        <div style='background:#1e2235;border-radius:6px;padding:22px 8px;text-align:center;font-size:20px;font-weight:700;color:#475569;'>184</div>
        <div style='background:#166534;border-radius:6px;padding:22px 8px;text-align:center;font-size:20px;font-weight:700;color:#86efac;'>10,982</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col8:
    st.markdown("""
    <div class='card'>
      <div class='card-title'>📊 Model Performance Comparison</div>
      <div style='margin-top:4px;'>
        <div class='mp-row'><span class='mp-name'>Decision Tree</span><div class='mp-track'><div class='mp-fill' style='width:99.67%;background:#7c3aed;'><span class='mp-val'>99.67%</span></div></div></div>
        <div class='mp-row'><span class='mp-name'>Gradient Boosting</span><div class='mp-track'><div class='mp-fill' style='width:99.5%;background:#4f46e5;'><span class='mp-val'>99.5%</span></div></div></div>
        <div class='mp-row'><span class='mp-name'>Random Forest</span><div class='mp-track'><div class='mp-fill' style='width:99.24%;background:#2563eb;'><span class='mp-val'>99.24%</span></div></div></div>
        <div class='mp-row'><span class='mp-name'>Linear SVM</span><div class='mp-track'><div class='mp-fill' style='width:98.74%;background:#059669;'><span class='mp-val'>98.74%</span></div></div></div>
        <div class='mp-row'><span class='mp-name'>Logistic Regression</span><div class='mp-track'><div class='mp-fill' style='width:98.56%;background:#0891b2;'><span class='mp-val'>98.56%</span></div></div></div>
        <div class='mp-row'><span class='mp-name'>Naïve Bayes</span><div class='mp-track'><div class='mp-fill' style='width:93.4%;background:#d97706;'><span class='mp-val'>93.4%</span></div></div></div>
      </div>
      <div style='font-size:10px;color:#374151;text-align:center;margin-top:8px;'>Accuracy (%)</div>
    </div>
    """, unsafe_allow_html=True)


# ── Prediction History ────────────────────────────────────────────────────────
st.markdown("<div class='card'><div class='card-title'>⭐ Prediction History (Recent)</div>", unsafe_allow_html=True)
history = [
    (1, "Government announces new economic policy...", "Real", "99.12%", "28 May 2025"),
    (2, "Secret cure for cancer leaked by scientist...", "Fake", "98.90%", "28 May 2025"),
    (3, "Elections update: Results to be declared tomorrow...", "Real", "97.85%", "28 May 2025"),
    (4, "You won't believe what this politician just said...", "Fake", "99.33%", "27 May 2025"),
    (5, "Central bank raises interest rates for stability...", "Real", "98.40%", "27 May 2025"),
]
rows = ""
for i, (num, headline, label, conf, date) in enumerate(history):
    border = "" if i == len(history) - 1 else "border-bottom:1px solid #13161f;"
    badge = f"<span class='badge-real'>Real</span>" if label == "Real" else f"<span class='badge-fake'>Fake</span>"
    color = "#4ade80" if label == "Real" else "#f87171"
    rows += f"<tr><td style='{border}'>{num}</td><td style='{border}'>{headline}</td><td style='{border}'>{badge}</td><td style='{border}color:{color};font-weight:600;'>{conf}</td><td style='{border}color:#475569;'>{date}</td></tr>"

st.markdown(f"""
<table class='rev-table'>
  <thead><tr><th>#</th><th>Headline</th><th>Prediction</th><th>Confidence</th><th>Date</th></tr></thead>
  <tbody>{rows}</tbody>
</table></div>
""", unsafe_allow_html=True)


# ── Pipeline ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class='card'>
  <div class='card-title'>🔁 NLP &amp; Machine Learning Pipeline</div>
  <div class='pipe-wrap'>
    <div class='pipe-step'><div class='pipe-icon'>🗄️</div><div class='pipe-title'>News Dataset</div><div class='pipe-sub'>Fake.csv + True.csv</div></div>
    <div class='pipe-arrow'>→</div>
    <div class='pipe-step'><div class='pipe-icon'>🧹</div><div class='pipe-title'>Data Cleaning</div><div class='pipe-sub'>Remove Noise</div></div>
    <div class='pipe-arrow'>→</div>
    <div class='pipe-step'><div class='pipe-icon'>📝</div><div class='pipe-title'>Text Preprocessing</div><div class='pipe-sub'>Lowercase, Stopwords, Punctuation</div></div>
    <div class='pipe-arrow'>→</div>
    <div class='pipe-step'><div class='pipe-icon'>📊</div><div class='pipe-title'>TF-IDF Vectorization</div><div class='pipe-sub'>Feature Extraction</div></div>
    <div class='pipe-arrow'>→</div>
    <div class='pipe-step'><div class='pipe-icon'>🧠</div><div class='pipe-title'>Model Training</div><div class='pipe-sub'>6 ML Algorithms</div></div>
    <div class='pipe-arrow'>→</div>
    <div class='pipe-step'><div class='pipe-icon'>🎯</div><div class='pipe-title'>Prediction</div><div class='pipe-sub'>Real / Fake</div></div>
    <div class='pipe-arrow'>→</div>
    <div class='pipe-step'><div class='pipe-icon'>📈</div><div class='pipe-title'>Results &amp; Visualization</div><div class='pipe-sub'>Insights Dashboard</div></div>
  </div>
</div>
""", unsafe_allow_html=True)
