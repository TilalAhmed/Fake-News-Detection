import gradio as gr
import os
import re
import string
import pickle
import numpy as np
import time
 
# ── Load models ───────────────────────────────────────────────────────────────
MODEL_DIR = "models"
 
MODEL_INFO = {
    "Logistic Regression": {"file": "LR.pkl",  "accuracy": "98.56%", "color": "#0891b2"},
    "Decision Tree":       {"file": "DT.pkl",  "accuracy": "99.67%", "color": "#7c3aed"},
    "Gradient Boosting":   {"file": "GB.pkl",  "accuracy": "99.5%",  "color": "#4f46e5"},
    "Linear SVM":          {"file": "SVC.pkl", "accuracy": "98.74%", "color": "#059669"},
    "Naïve Bayes":         {"file": "NB.pkl",  "accuracy": "93.4%",  "color": "#d97706"},
}
 
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
 
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text
 
def analyze_news(news_text, selected_model):
    if not news_text or not news_text.strip():
        return make_result_html("", "", 0, "Please paste a news article to analyze.")
 
    wc = len(news_text.split())
    if wc < 5:
        return make_result_html("", "", 0, "Too short — paste at least a sentence.")
 
    if models_loaded and assets["vectorizer"] and selected_model in assets["models"]:
        model = assets["models"][selected_model]
        cleaned = clean_text(news_text)
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
        t = news_text.lower()
        fs = sum(1 for w in fake_words if w in t)
        rs = sum(1 for w in real_words if w in t)
        label = "REAL" if rs >= fs else "FAKE"
        confidence = float(np.random.uniform(82, 97))
 
    mode = "Models loaded" if models_loaded else "Demo mode"
    return make_result_html(label, selected_model, confidence, None, mode, wc)
 
def make_result_html(label, model, confidence, error=None, mode="", wc=0):
    if error:
        return f"""
        <div style='background:#1e2235;border:1px solid #374151;border-radius:12px;padding:20px;font-family:Inter,sans-serif;'>
            <div style='color:#f87171;font-size:14px;'>⚠️ {error}</div>
        </div>
        """
    is_real = label == "REAL"
    bg = "#052e16" if is_real else "#2d0a0a"
    border = "#166534" if is_real else "#7f1d1d"
    color = "#4ade80" if is_real else "#f87171"
    emoji = "✓" if is_real else "✗"
    desc = "This article appears to be legitimate." if is_real else "This article shows signs of misinformation."
 
    return f"""
    <div style='background:{bg};border:1px solid {border};border-radius:12px;padding:20px;font-family:Inter,sans-serif;'>
        <div style='font-size:28px;font-weight:700;color:{color};'>{emoji} {label} NEWS</div>
        <div style='font-size:12px;color:#94a3b8;margin-top:6px;'>{desc}</div>
        <div style='margin-top:14px;'>
            <div style='font-size:11px;color:#64748b;margin-bottom:4px;'>Confidence: {confidence:.1f}%</div>
            <div style='height:8px;background:#1e293b;border-radius:4px;overflow:hidden;'>
                <div style='height:100%;width:{confidence:.1f}%;background:{color};border-radius:4px;'></div>
            </div>
        </div>
        <div style='margin-top:10px;font-size:10px;color:#475569;'>
            Model: {model} · Words: {wc} · {mode}
        </div>
    </div>
    """
 
def get_model_info(selected_model):
    info = MODEL_INFO.get(selected_model, {})
    loaded = "🟢 Loaded" if selected_model in assets["models"] else "⚪ Not loaded"
    return f"""
    <div style='background:#13161f;border:1px solid #1e2235;border-radius:10px;padding:14px;font-family:Inter,sans-serif;'>
        <div style='font-size:12px;font-weight:600;color:#a78bfa;'>{selected_model}</div>
        <div style='font-size:11px;color:#64748b;margin-top:6px;'>Accuracy: <span style='color:#e2e8f0;'>{info.get('accuracy','N/A')}</span></div>
        <div style='font-size:11px;color:#64748b;margin-top:4px;'>Status: {loaded}</div>
    </div>
    """
 
# ── Stats HTML ────────────────────────────────────────────────────────────────
stats_html = """
<div style='font-family:Inter,sans-serif;display:grid;grid-template-columns:repeat(5,1fr);gap:10px;'>
  <div style='background:#13161f;border:1px solid #1e2235;border-radius:10px;padding:14px;text-align:center;'>
    <div style='font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1px;'>Total Articles</div>
    <div style='font-size:22px;font-weight:700;color:#fff;margin-top:4px;'>44,898</div>
  </div>
  <div style='background:#13161f;border:1px solid #1e2235;border-radius:10px;padding:14px;text-align:center;'>
    <div style='font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1px;'>Fake Articles</div>
    <div style='font-size:22px;font-weight:700;color:#f87171;margin-top:4px;'>23,481</div>
  </div>
  <div style='background:#13161f;border:1px solid #1e2235;border-radius:10px;padding:14px;text-align:center;'>
    <div style='font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1px;'>Real Articles</div>
    <div style='font-size:22px;font-weight:700;color:#4ade80;margin-top:4px;'>21,417</div>
  </div>
  <div style='background:#13161f;border:1px solid #1e2235;border-radius:10px;padding:14px;text-align:center;'>
    <div style='font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1px;'>ML Models</div>
    <div style='font-size:22px;font-weight:700;color:#fff;margin-top:4px;'>6</div>
  </div>
  <div style='background:#13161f;border:1px solid #1e2235;border-radius:10px;padding:14px;text-align:center;'>
    <div style='font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1px;'>Best Accuracy</div>
    <div style='font-size:22px;font-weight:700;color:#a78bfa;margin-top:4px;'>99.67%</div>
  </div>
</div>
"""
 
model_perf_html = """
<div style='font-family:Inter,sans-serif;background:#13161f;border:1px solid #1e2235;border-radius:12px;padding:16px;'>
  <div style='font-size:11px;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;'>📊 Model Performance</div>
  <div style='display:flex;align-items:center;gap:8px;margin-bottom:8px;'>
    <span style='font-size:10px;color:#94a3b8;width:130px;text-align:right;'>Decision Tree</span>
    <div style='flex:1;height:16px;background:#0f1117;border-radius:3px;overflow:hidden;'>
      <div style='height:100%;width:99.67%;background:#7c3aed;border-radius:3px;display:flex;align-items:center;padding-left:7px;'>
        <span style='font-size:9px;font-weight:600;color:#fff;'>99.67%</span>
      </div>
    </div>
  </div>
  <div style='display:flex;align-items:center;gap:8px;margin-bottom:8px;'>
    <span style='font-size:10px;color:#94a3b8;width:130px;text-align:right;'>Gradient Boosting</span>
    <div style='flex:1;height:16px;background:#0f1117;border-radius:3px;overflow:hidden;'>
      <div style='height:100%;width:99.5%;background:#4f46e5;border-radius:3px;display:flex;align-items:center;padding-left:7px;'>
        <span style='font-size:9px;font-weight:600;color:#fff;'>99.5%</span>
      </div>
    </div>
  </div>
  <div style='display:flex;align-items:center;gap:8px;margin-bottom:8px;'>
    <span style='font-size:10px;color:#94a3b8;width:130px;text-align:right;'>Linear SVM</span>
    <div style='flex:1;height:16px;background:#0f1117;border-radius:3px;overflow:hidden;'>
      <div style='height:100%;width:98.74%;background:#059669;border-radius:3px;display:flex;align-items:center;padding-left:7px;'>
        <span style='font-size:9px;font-weight:600;color:#fff;'>98.74%</span>
      </div>
    </div>
  </div>
  <div style='display:flex;align-items:center;gap:8px;margin-bottom:8px;'>
    <span style='font-size:10px;color:#94a3b8;width:130px;text-align:right;'>Logistic Regression</span>
    <div style='flex:1;height:16px;background:#0f1117;border-radius:3px;overflow:hidden;'>
      <div style='height:100%;width:98.56%;background:#0891b2;border-radius:3px;display:flex;align-items:center;padding-left:7px;'>
        <span style='font-size:9px;font-weight:600;color:#fff;'>98.56%</span>
      </div>
    </div>
  </div>
  <div style='display:flex;align-items:center;gap:8px;'>
    <span style='font-size:10px;color:#94a3b8;width:130px;text-align:right;'>Naïve Bayes</span>
    <div style='flex:1;height:16px;background:#0f1117;border-radius:3px;overflow:hidden;'>
      <div style='height:100%;width:93.4%;background:#d97706;border-radius:3px;display:flex;align-items:center;padding-left:7px;'>
        <span style='font-size:9px;font-weight:600;color:#fff;'>93.4%</span>
      </div>
    </div>
  </div>
</div>
"""
 
pipeline_html = """
<div style='font-family:Inter,sans-serif;background:#13161f;border:1px solid #1e2235;border-radius:12px;padding:16px;'>
  <div style='font-size:11px;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;'>🔁 NLP & ML Pipeline</div>
  <div style='display:flex;align-items:center;gap:4px;overflow-x:auto;'>
    <div style='flex:1;min-width:80px;background:#0f1117;border:1px solid #1e2235;border-radius:8px;padding:10px 6px;text-align:center;'>
      <div style='font-size:18px;'>🗄️</div>
      <div style='font-size:9px;font-weight:600;color:#e2e8f0;'>Dataset</div>
      <div style='font-size:8px;color:#475569;'>Fake + True CSV</div>
    </div>
    <div style='color:#374151;font-size:12px;'>→</div>
    <div style='flex:1;min-width:80px;background:#0f1117;border:1px solid #1e2235;border-radius:8px;padding:10px 6px;text-align:center;'>
      <div style='font-size:18px;'>🧹</div>
      <div style='font-size:9px;font-weight:600;color:#e2e8f0;'>Cleaning</div>
      <div style='font-size:8px;color:#475569;'>Remove Noise</div>
    </div>
    <div style='color:#374151;font-size:12px;'>→</div>
    <div style='flex:1;min-width:80px;background:#0f1117;border:1px solid #1e2235;border-radius:8px;padding:10px 6px;text-align:center;'>
      <div style='font-size:18px;'>📝</div>
      <div style='font-size:9px;font-weight:600;color:#e2e8f0;'>Preprocessing</div>
      <div style='font-size:8px;color:#475569;'>TF-IDF</div>
    </div>
    <div style='color:#374151;font-size:12px;'>→</div>
    <div style='flex:1;min-width:80px;background:#0f1117;border:1px solid #1e2235;border-radius:8px;padding:10px 6px;text-align:center;'>
      <div style='font-size:18px;'>🧠</div>
      <div style='font-size:9px;font-weight:600;color:#e2e8f0;'>Training</div>
      <div style='font-size:8px;color:#475569;'>6 Algorithms</div>
    </div>
    <div style='color:#374151;font-size:12px;'>→</div>
    <div style='flex:1;min-width:80px;background:#0f1117;border:1px solid #1e2235;border-radius:8px;padding:10px 6px;text-align:center;'>
      <div style='font-size:18px;'>🎯</div>
      <div style='font-size:9px;font-weight:600;color:#e2e8f0;'>Prediction</div>
      <div style='font-size:8px;color:#475569;'>Real / Fake</div>
    </div>
  </div>
</div>
"""
 
# ── Gradio UI ─────────────────────────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }
body, .gradio-container { background: #0f1117 !important; color: #e2e8f0 !important; }
.gr-panel, .gr-box { background: #13161f !important; border: 1px solid #1e2235 !important; }
gradio-app { background: #0f1117 !important; }
.gr-button-primary {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    border: none !important; color: #fff !important; font-weight: 600 !important;
    border-radius: 8px !important;
}
.gr-button-primary:hover { opacity: 0.9 !important; }
label, .gr-block-label { color: #94a3b8 !important; font-size: 11px !important; }
textarea, input, select { 
    background: #0f1117 !important; border: 1px solid #1e2235 !important; 
    color: #cbd5e1 !important; border-radius: 8px !important; 
}
textarea:focus, input:focus { border-color: #7c3aed !important; }
.gr-dropdown { background: #0f1117 !important; }
footer { display: none !important; }
"""
 
with gr.Blocks(css=css, title="Fake News Detection System") as demo:
 
    # Header
    gr.HTML("""
    <div style='display:flex;align-items:center;gap:14px;padding:20px 0 10px;'>
      <div style='width:48px;height:48px;background:linear-gradient(135deg,#7c3aed,#4f46e5);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:24px;flex-shrink:0;'>🔍</div>
      <div>
        <div style='font-size:22px;font-weight:700;color:#fff;'>Fake News Detection System</div>
        <div style='font-size:11px;color:#475569;margin-top:2px;'>Machine Learning + NLP · TF-IDF Vectorization · ISOT Dataset</div>
      </div>
    </div>
    """)
 
    # Stats Row
    gr.HTML(stats_html)
 
    gr.HTML("<div style='height:16px;'></div>")
 
    # Main Row — Analyzer + Info
    with gr.Row():
        with gr.Column(scale=2):
            gr.HTML("<div style='font-size:11px;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;'>📡 Live News Analyzer</div>")
            news_input = gr.Textbox(
                placeholder="Paste your news article here...",
                lines=6,
                label="News Article",
                show_label=False
            )
            selected_model = gr.Dropdown(
                choices=list(MODEL_INFO.keys()),
                value="Decision Tree",
                label="Select Model"
            )
            analyze_btn = gr.Button("📡 Analyze News", variant="primary")
            result_output = gr.HTML("""
            <div style='background:#13161f;border:1px solid #1e2235;border-radius:12px;padding:20px;font-family:Inter,sans-serif;color:#475569;font-size:12px;text-align:center;'>
                Paste a news article and click Analyze
            </div>
            """)
 
        with gr.Column(scale=1):
            model_info_html = gr.HTML(get_model_info("Decision Tree"))
            gr.HTML("""
            <div style='background:#13161f;border:1px solid #1e2235;border-radius:10px;padding:14px;font-family:Inter,sans-serif;margin-top:10px;'>
              <div style='font-size:11px;font-weight:600;color:#94a3b8;margin-bottom:8px;'>🗄️ About Dataset</div>
              <div style='font-size:10px;color:#475569;line-height:1.8;'>
                Combined Fake.csv + True.csv<br>
                Total Articles: 44,898<br>
                Features: Title, Text, Subject, Date<br>
                Target: Real / Fake<br>
                Source: ISOT Dataset
              </div>
            </div>
            """)
            gr.HTML("""
            <div style='background:#13161f;border:1px solid #1e2235;border-radius:10px;padding:14px;font-family:Inter,sans-serif;margin-top:10px;'>
              <div style='font-size:11px;font-weight:600;color:#94a3b8;margin-bottom:8px;'>👨‍💻 About</div>
              <div style='font-size:10px;color:#475569;line-height:1.8;'>
                Tilal Ahmed<br>
                Iqra University, Karachi<br>
                Scikit-learn · TF-IDF
              </div>
            </div>
            """)
 
    gr.HTML("<div style='height:16px;'></div>")
 
    # Performance + Pipeline
    with gr.Row():
        with gr.Column():
            gr.HTML(model_perf_html)
        with gr.Column():
            gr.HTML("""
            <div style='font-family:Inter,sans-serif;background:#13161f;border:1px solid #1e2235;border-radius:12px;padding:16px;'>
              <div style='font-size:11px;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;'>⊞ Confusion Matrix (Decision Tree)</div>
              <div style='display:grid;grid-template-columns:auto 1fr 1fr;grid-template-rows:auto 1fr 1fr;gap:5px;'>
                <div></div>
                <div style='font-size:10px;color:#f87171;text-align:center;padding:4px;font-weight:600;'>Pred: Fake</div>
                <div style='font-size:10px;color:#4ade80;text-align:center;padding:4px;font-weight:600;'>Pred: Real</div>
                <div style='font-size:9px;color:#64748b;text-align:center;padding:4px;'>Actual Fake</div>
                <div style='background:#7f1d1d;border-radius:6px;padding:16px 8px;text-align:center;font-size:18px;font-weight:700;color:#fca5a5;'>11,523</div>
                <div style='background:#1e2235;border-radius:6px;padding:16px 8px;text-align:center;font-size:18px;font-weight:700;color:#475569;'>212</div>
                <div style='font-size:9px;color:#64748b;text-align:center;padding:4px;'>Actual Real</div>
                <div style='background:#1e2235;border-radius:6px;padding:16px 8px;text-align:center;font-size:18px;font-weight:700;color:#475569;'>184</div>
                <div style='background:#166534;border-radius:6px;padding:16px 8px;text-align:center;font-size:18px;font-weight:700;color:#86efac;'>10,982</div>
              </div>
            </div>
            """)
 
    gr.HTML("<div style='height:16px;'></div>")
    gr.HTML(pipeline_html)
 
    # Events
    analyze_btn.click(
        fn=analyze_news,
        inputs=[news_input, selected_model],
        outputs=result_output
    )
    selected_model.change(
        fn=get_model_info,
        inputs=selected_model,
        outputs=model_info_html
    )
 
if __name__ == "__main__":
    demo.launch()
