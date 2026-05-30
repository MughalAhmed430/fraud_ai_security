"""
dashboard/app.py  —  Financial AI Security Dashboard
=====================================================
A professional dark-theme dashboard telling the complete story:
  Problem → Dataset → Model → Attack → Defense → Live Demo

Run:  streamlit run dashboard/app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib, os, sys

sys.path.insert(0, ".")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield AI — Security Research",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System ─────────────────────────────────────────────────────────────
COLORS = {
    "bg":        "#0f1117",
    "surface":   "#1a1a2e",
    "surface2":  "#16213e",
    "border":    "#2d2d44",
    "cyan":      "#00E5FF",
    "green":     "#69FF47",
    "red":       "#FF4B4B",
    "orange":    "#FF9800",
    "purple":    "#B388FF",
    "text":      "#E0E0E0",
    "muted":     "#888899",
}

st.markdown(f"""
<style>
  /* ── Global ── */
  html, body, [data-testid="stApp"] {{
      background-color: {COLORS['bg']};
      color: {COLORS['text']};
      font-family: 'Inter', 'Segoe UI', sans-serif;
  }}
  [data-testid="stSidebar"] {{
      background-color: {COLORS['surface']} !important;
      border-right: 1px solid {COLORS['border']};
  }}
  [data-testid="stSidebar"] * {{ color: {COLORS['text']} !important; }}

  /* ── Metric cards ── */
  [data-testid="metric-container"] {{
      background: {COLORS['surface']};
      border: 1px solid {COLORS['border']};
      border-radius: 12px;
      padding: 1rem 1.2rem;
  }}
  [data-testid="stMetricValue"]  {{ color: {COLORS['cyan']} !important; font-size:1.8rem !important; font-weight:800; }}
  [data-testid="stMetricLabel"]  {{ color: {COLORS['muted']} !important; font-size:0.8rem !important; text-transform:uppercase; letter-spacing:.05em; }}
  [data-testid="stMetricDelta"]  {{ font-size:0.85rem !important; }}

  /* ── Tabs ── */
  [data-testid="stTab"] {{
      background: transparent;
      border-bottom: 2px solid {COLORS['border']};
      color: {COLORS['muted']};
  }}
  [aria-selected="true"] {{
      border-bottom: 2px solid {COLORS['cyan']} !important;
      color: {COLORS['cyan']} !important;
  }}

  /* ── Buttons ── */
  [data-testid="baseButton-primary"] {{
      background: linear-gradient(135deg, #00B4D8, #0077B6) !important;
      border: none !important;
      border-radius: 8px !important;
      font-weight: 700 !important;
      letter-spacing: .03em;
  }}
  [data-testid="baseButton-secondary"] {{
      background: {COLORS['surface']} !important;
      border: 1px solid {COLORS['border']} !important;
      border-radius: 8px !important;
  }}

  /* ── Divider ── */
  hr {{ border-color: {COLORS['border']} !important; }}

  /* ── Expander ── */
  [data-testid="stExpander"] {{
      background: {COLORS['surface']};
      border: 1px solid {COLORS['border']};
      border-radius: 10px;
  }}

  /* ── Slider ── */
  [data-testid="stSlider"] > div > div {{ background: {COLORS['cyan']} !important; }}

  /* ── File uploader ── */
  [data-testid="stFileUploader"] {{
      background: {COLORS['surface']};
      border: 1px dashed {COLORS['border']};
      border-radius: 10px;
  }}

  /* ── Custom cards ── */
  .hero-title {{
      font-size: 2.8rem; font-weight: 900; letter-spacing: -.02em;
      background: linear-gradient(135deg, {COLORS['cyan']}, {COLORS['purple']});
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      line-height: 1.1; margin-bottom: .4rem;
  }}
  .hero-sub {{
      font-size: 1.15rem; color: {COLORS['muted']}; margin-bottom: 2rem;
  }}
  .section-title {{
      font-size: 1.4rem; font-weight: 700; color: {COLORS['text']};
      border-left: 4px solid {COLORS['cyan']};
      padding-left: .75rem; margin-bottom: 1rem;
  }}
  .card {{
      background: {COLORS['surface']};
      border: 1px solid {COLORS['border']};
      border-radius: 14px;
      padding: 1.4rem 1.6rem;
      margin-bottom: 1rem;
      height: 100%;
  }}
  .card-red   {{ border-top: 3px solid {COLORS['red']}; }}
  .card-green {{ border-top: 3px solid {COLORS['green']}; }}
  .card-cyan  {{ border-top: 3px solid {COLORS['cyan']}; }}
  .card-purple{{ border-top: 3px solid {COLORS['purple']}; }}
  .card-orange{{ border-top: 3px solid {COLORS['orange']}; }}
  .card h3 {{ font-size:1.05rem; font-weight:700; margin-bottom:.6rem; }}
  .card p  {{ font-size:.9rem; color:{COLORS['muted']}; line-height:1.6; }}

  .stat-big {{
      font-size: 3rem; font-weight: 900; line-height: 1;
      color: {COLORS['cyan']};
  }}
  .stat-label {{
      font-size: .8rem; color: {COLORS['muted']};
      text-transform: uppercase; letter-spacing: .05em;
  }}
  .badge {{
      display: inline-block; padding: 3px 12px; border-radius: 20px;
      font-size: .75rem; font-weight: 700; margin: 2px;
  }}
  .badge-red    {{ background: rgba(255,75,75,.2);   color:{COLORS['red']};    border:1px solid {COLORS['red']}; }}
  .badge-green  {{ background: rgba(105,255,71,.2);  color:{COLORS['green']};  border:1px solid {COLORS['green']}; }}
  .badge-cyan   {{ background: rgba(0,229,255,.2);   color:{COLORS['cyan']};   border:1px solid {COLORS['cyan']}; }}
  .badge-orange {{ background: rgba(255,152,0,.2);   color:{COLORS['orange']}; border:1px solid {COLORS['orange']}; }}
  .badge-purple {{ background: rgba(179,136,255,.2); color:{COLORS['purple']}; border:1px solid {COLORS['purple']}; }}

  .formula-box {{
      background: #0d1117; border: 1px solid {COLORS['border']};
      border-radius: 8px; padding: 1rem 1.2rem;
      font-family: 'Fira Code', 'Consolas', monospace;
      font-size: .9rem; color: {COLORS['cyan']};
      margin: .5rem 0;
  }}
  .problem-stat {{
      text-align: center; padding: 1.5rem;
      background: {COLORS['surface']};
      border: 1px solid {COLORS['border']};
      border-radius: 12px;
  }}
  .timeline-step {{
      display: flex; align-items: flex-start;
      gap: 1rem; margin-bottom: 1.2rem;
  }}
  .step-num {{
      background: linear-gradient(135deg, {COLORS['cyan']}, {COLORS['purple']});
      color: {COLORS['bg']}; font-weight: 900;
      width: 32px; height: 32px; border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: .9rem; flex-shrink: 0;
  }}
  .step-content h4 {{ margin:0 0 .2rem; font-size:.95rem; }}
  .step-content p  {{ margin:0; font-size:.85rem; color:{COLORS['muted']}; }}
</style>
""", unsafe_allow_html=True)


# ── Load artefacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        import tensorflow as tf
        from models.train_model import FocalLoss
        custom_obj = {"FocalLoss": FocalLoss}
        baseline = tf.keras.models.load_model("models/fraud_model.h5",    custom_objects=custom_obj)
        defended = tf.keras.models.load_model("defense/defended_model.h5", custom_objects=custom_obj)
        return baseline, defended
    except Exception:
        return None, None

@st.cache_data
def load_data():
    try:
        X_test       = np.load("models/X_test.npy")
        y_test       = np.load("models/y_test.npy")
        X_adv_fgsm   = np.load("models/X_adv_fgsm.npy")
        X_adv_pgd    = np.load("models/X_adv_pgd.npy")
        X_orig_fraud = np.load("models/X_orig_fraud.npy")
        y_fraud      = np.load("models/y_fraud.npy")
        summary      = joblib.load("models/pipeline_summary.pkl")
        threshold    = joblib.load("models/optimal_threshold.pkl")
        return X_test, y_test, X_adv_fgsm, X_adv_pgd, X_orig_fraud, y_fraud, summary, threshold
    except Exception:
        return None, None, None, None, None, None, None, 0.5

baseline_model, defended_model = load_models()
X_test, y_test, X_adv_fgsm, X_adv_pgd, X_orig_fraud, y_fraud, summary, threshold = load_data()
models_loaded = baseline_model is not None
data_loaded   = X_test is not None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0;'>
      <div style='font-size:2.5rem;'>🛡️</div>
      <div style='font-size:1.2rem; font-weight:800; color:#00E5FF;'>FraudShield AI</div>
      <div style='font-size:.75rem; color:#888899; margin-top:.2rem;'>
        Adversarial Security Research
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigation",
        ["Overview", "The Problem", "Our Model", "Under Attack",
         "Defense", "Live Demo", "How It Works"],
        label_visibility="collapsed"
    )

    st.divider()

    if models_loaded:
        st.success("Models loaded")
        if data_loaded and summary:
            st.markdown(f"""
            <div style='font-size:.8rem; color:#888;'>
              Recall: <b style='color:#69FF47'>{summary['baseline']['recall']*100:.1f}%</b><br>
              PR-AUC: <b style='color:#00E5FF'>{summary['baseline']['pr_auc']:.3f}</b><br>
              FGSM evasion: <b style='color:#FF4B4B'>{summary['fgsm']['evasion_rate']:.1f}%</b>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("Run pipeline first:\n`python run_pipeline.py`")

    st.divider()
    st.markdown(f"<div style='font-size:.72rem; color:#555; text-align:center;'>Threshold: {threshold:.4f}</div>",
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "Overview":
    st.markdown('<div class="hero-title">FraudShield AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Adversarial Machine Learning Research on Financial Fraud Detection Systems</div>',
        unsafe_allow_html=True
    )

    # ── Key stats row ─────────────────────────────────────────────────────────
    if data_loaded and summary:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Transactions", "284,807", "Real Kaggle data")
        c2.metric("Fraud Cases", "492", "0.17% of all")
        c3.metric("Model Recall", f"{summary['baseline']['recall']*100:.1f}%", "Frauds caught")
        c4.metric("PR-AUC", f"{summary['baseline']['pr_auc']:.3f}", "Primary metric")
        c5.metric("FGSM Evasion", f"{summary['fgsm']['evasion_rate']:.1f}%", "Attack success rate")

    st.divider()

    # ── Three-column summary ──────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="card card-cyan">
          <h3>🎯 The Problem</h3>
          <p>Credit card fraud costs the world <b style='color:#00E5FF'>$33 billion/year</b>.
          Banks use AI to detect fraud in real-time — but these AI models have a hidden
          vulnerability: they can be fooled by <b>mathematically crafted perturbations</b>
          invisible to the human eye.</p>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card card-red">
          <h3>⚔️ The Threat</h3>
          <p>Using <b>FGSM</b> and <b>PGD</b> adversarial attacks, we demonstrate that a
          fraud transaction can be modified by tiny amounts — making the AI classify it as
          <b style='color:#FF4B4B'>completely normal</b>.
          Evasion rates can reach <b>90%+</b> on a vulnerable model.</p>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card card-green">
          <h3>🛡️ The Solution</h3>
          <p>We apply two defenses: <b>Adversarial Training</b> (training on attacked examples)
          and <b>Feature Squeezing</b> (destroying perturbation precision).
          Our defended model recovers <b style='color:#69FF47'>70-80% recall</b>
          even under the strongest attacks.</p>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Pipeline flow ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">End-to-End Pipeline</div>', unsafe_allow_html=True)

    steps = [
        ("1", "Data Ingestion", "284,807 real credit card transactions from Kaggle. Only 492 (0.17%) are fraud — extreme class imbalance handled with SMOTE."),
        ("2", "Model Training", "Deep Residual Neural Network with Focal Loss + class weights. Optimised for PR-AUC, not just accuracy. Threshold tuned for maximum F1."),
        ("3", "Adversarial Attacks", "FGSM (1-step gradient) and PGD (40-step iterative) attacks from IBM ART library. Target: make fraud transactions look normal to the AI."),
        ("4", "Defense", "Adversarial Training embeds attack examples into training. Feature Squeezing destroys tiny perturbations at inference time."),
        ("5", "Evaluation", "Side-by-side comparison: vulnerable vs defended model across clean data, FGSM attacks, and PGD attacks."),
    ]

    for num, title, desc in steps:
        st.markdown(f"""
        <div class="timeline-step">
          <div class="step-num">{num}</div>
          <div class="step-content">
            <h4>{title}</h4>
            <p>{desc}</p>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Architecture diagram ──────────────────────────────────────────────────
    if os.path.exists("diagrams/architecture.png"):
        st.divider()
        st.markdown('<div class="section-title">System Architecture</div>', unsafe_allow_html=True)
        st.image("diagrams/architecture.png", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: THE PROBLEM
# ══════════════════════════════════════════════════════════════════════════════

elif page == "The Problem":
    st.markdown('<div class="hero-title">The Problem</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Why fraud detection AI is vulnerable — and why it matters</div>',
                unsafe_allow_html=True)

    # ── Problem stats ─────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    for col, stat, label in zip(
        [c1, c2, c3, c4],
        ["$33B", "1 in 578", "0.17%", "< 50ms"],
        ["Annual card fraud losses", "Transactions are fraudulent", "Fraud rate in dataset", "Bank decision window"]
    ):
        col.markdown(f"""
        <div class="problem-stat">
          <div class="stat-big">{stat}</div>
          <div class="stat-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown('<div class="section-title">The Class Imbalance Problem</div>', unsafe_allow_html=True)

        # Imbalance chart
        fig = go.Figure(go.Bar(
            x=["Normal Transactions", "Fraud Transactions"],
            y=[284315, 492],
            marker_color=[COLORS["cyan"], COLORS["red"]],
            text=["284,315 (99.83%)", "492 (0.17%)"],
            textposition="outside",
            textfont=dict(color="white", size=12),
        ))
        fig.update_layout(
            title="Dataset Class Distribution",
            paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["surface"],
            font=dict(color=COLORS["text"]),
            yaxis=dict(type="log", title="Count (log scale)", gridcolor=COLORS["border"]),
            xaxis=dict(gridcolor=COLORS["border"]),
            height=320, margin=dict(t=40, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class="card card-orange">
          <h3>⚠️ Why Accuracy Is Misleading</h3>
          <p>A model that predicts <b>"Normal"</b> for <i>every single transaction</i>
          would achieve <b style='color:{COLORS["orange"]}'>99.83% accuracy</b> —
          while missing 100% of all fraud.<br><br>
          This is why we use <b>PR-AUC</b> (Precision-Recall Area Under Curve)
          as our primary metric, and optimise the decision threshold for maximum F1.</p>
        </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-title">What Is an Adversarial Attack?</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card card-red" style='margin-bottom:.8rem;'>
          <h3>The Core Vulnerability</h3>
          <p>AI models classify inputs using a mathematical decision boundary.
          Adversarial attacks find the <b>minimal perturbation</b> needed to push
          an input across that boundary — from "Fraud" to "Normal".</p>
        </div>
        <div class="card" style='margin-bottom:.8rem;'>
          <h3>Real-World Threat Scenario</h3>
          <p>
          1. Criminal knows the fraud detection model (white-box attack)<br>
          2. Calculates tiny changes to make each fraudulent transaction look normal<br>
          3. Changes are imperceptible — just <b style='color:{COLORS["cyan"]}'>±0.1 units</b> per feature<br>
          4. AI model is fooled → transaction approved → money stolen
          </p>
        </div>
        """, unsafe_allow_html=True)

        # SMOTE explanation chart
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name="Before SMOTE", x=["Normal", "Fraud"],
            y=[227452, 394], marker_color=[COLORS["cyan"], COLORS["red"]],
            opacity=0.6,
        ))
        fig2.add_trace(go.Bar(
            name="After SMOTE", x=["Normal", "Fraud"],
            y=[227452, 227452], marker_color=[COLORS["cyan"], COLORS["green"]],
            opacity=0.9,
        ))
        fig2.update_layout(
            title="SMOTE Balancing Effect (Training Set)",
            barmode="group",
            paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["surface"],
            font=dict(color=COLORS["text"]),
            yaxis=dict(gridcolor=COLORS["border"]),
            height=280, margin=dict(t=40, b=20),
            legend=dict(bgcolor=COLORS["surface"]),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OUR MODEL
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Our Model":
    st.markdown('<div class="hero-title">Our Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Deep Residual Network trained with Focal Loss for best-in-class fraud detection</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section-title">Architecture</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card card-cyan">
          <h3>Deep Residual Network</h3>
          <p>Three residual blocks with skip connections prevent vanishing gradients
          and allow the model to learn <b>corrections on top of identity mappings</b>.</p>
        </div>
        """, unsafe_allow_html=True)

        # Architecture flow
        arch_steps = [
            ("Input", "29 features (V1-V28 + Amount)", COLORS["cyan"]),
            ("Block 1", "Dense(128) → BN → LeakyReLU → Skip", COLORS["purple"]),
            ("Block 2", "Dense(64) → BN → LeakyReLU → Skip", COLORS["purple"]),
            ("Block 3", "Dense(32) → BN → LeakyReLU", COLORS["purple"]),
            ("Output", "Dense(1, Sigmoid) → Fraud Probability", COLORS["green"]),
        ]

        for name, detail, color in arch_steps:
            st.markdown(f"""
            <div style='background:{COLORS["surface2"]}; border:1px solid {COLORS["border"]};
                        border-left:4px solid {color}; border-radius:8px;
                        padding:.6rem 1rem; margin-bottom:.4rem;'>
              <b style='color:{color}'>{name}</b>
              <span style='color:{COLORS["muted"]}; font-size:.85rem;'> — {detail}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="formula-box">
FL(p) = -α · (1 - p)^γ · log(p)

γ = 2.0   (focus on hard misclassifications)
α = 0.25  (weight for fraud class)
        </div>
        """, unsafe_allow_html=True)
        st.caption("Focal Loss — focuses training on hard fraud cases, ignores easy negatives")

    with col2:
        st.markdown('<div class="section-title">Performance Metrics</div>', unsafe_allow_html=True)

        if data_loaded and summary:
            b = summary["baseline"]

            # Gauge chart for recall
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=b["recall"] * 100,
                number={"suffix": "%", "font": {"size": 40, "color": COLORS["green"]}},
                delta={"reference": 80, "valueformat": ".1f"},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": COLORS["muted"]},
                    "bar": {"color": COLORS["green"]},
                    "bgcolor": COLORS["surface"],
                    "bordercolor": COLORS["border"],
                    "steps": [
                        {"range": [0, 60],  "color": "#1a0a0a"},
                        {"range": [60, 80], "color": "#1a1500"},
                        {"range": [80, 100],"color": "#0a1a0a"},
                    ],
                    "threshold": {
                        "line": {"color": COLORS["cyan"], "width": 3},
                        "thickness": 0.8, "value": 80
                    },
                },
                title={"text": "Fraud Recall", "font": {"color": COLORS["text"]}},
            ))
            fig_gauge.update_layout(
                paper_bgcolor=COLORS["bg"], font=dict(color=COLORS["text"]),
                height=280, margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Metrics table
            metrics = {
                "Accuracy":  f"{b['accuracy']*100:.2f}%",
                "Precision": f"{b['precision']*100:.2f}%",
                "Recall":    f"{b['recall']*100:.2f}%",
                "F1 Score":  f"{b['f1']*100:.2f}%",
                "ROC-AUC":   f"{b['roc_auc']:.4f}",
                "PR-AUC":    f"{b['pr_auc']:.4f}",
                "Threshold": f"{b['threshold']:.4f}",
            }
            df_metrics = pd.DataFrame(
                [{"Metric": k, "Score": v} for k, v in metrics.items()]
            )
            st.dataframe(
                df_metrics, hide_index=True, use_container_width=True,
                column_config={
                    "Metric": st.column_config.TextColumn("Metric"),
                    "Score":  st.column_config.TextColumn("Score"),
                }
            )
        else:
            st.info("Run pipeline to see metrics.")

    st.divider()
    tab1, tab2, tab3 = st.tabs(["Training History", "Precision-Recall Curve", "Confusion Matrix"])

    with tab1:
        if os.path.exists("dashboard/assets/training_history.png"):
            st.image("dashboard/assets/training_history.png", use_container_width=True)

    with tab2:
        if os.path.exists("dashboard/assets/pr_curve.png"):
            st.image("dashboard/assets/pr_curve.png", use_container_width=True)
        st.caption("PR-AUC tells us how well the model separates fraud from normal across all thresholds. "
                   "Random classifier = flat line at 0.17% (fraud rate). Our model should be much higher.")

    with tab3:
        if os.path.exists("dashboard/assets/cm_baseline.png"):
            col_cm, col_exp = st.columns([1, 1])
            with col_cm:
                st.image("dashboard/assets/cm_baseline.png", use_container_width=True)
            with col_exp:
                st.markdown(f"""
                <div class="card" style='margin-top:1rem;'>
                  <h3>Reading the Matrix</h3>
                  <p>
                  <b style='color:{COLORS["green"]}'>True Positives (TP)</b> — Fraud correctly caught<br>
                  <b style='color:{COLORS["green"]}'>True Negatives (TN)</b> — Normal correctly passed<br>
                  <b style='color:{COLORS["red"]}'>False Negatives (FN)</b> — Fraud missed (most costly!)<br>
                  <b style='color:{COLORS["orange"]}'>False Positives (FP)</b> — Normal flagged as fraud (customer friction)
                  </p>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: UNDER ATTACK
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Under Attack":
    st.markdown('<div class="hero-title">Under Attack</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">How adversarial attacks systematically destroy fraud detection performance</div>',
                unsafe_allow_html=True)

    if data_loaded and summary:
        # ── Attack summary cards ──────────────────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="card card-orange">
              <h3>⚡ FGSM Attack (Fast Gradient Sign Method)</h3>
              <p>One-shot gradient attack. Takes a single step in the direction
              that maximises the model's mistake.</p>
              <br>
              <b style='color:{COLORS["cyan"]}'>Before: </b>
              <b style='color:{COLORS["green"]}'>{summary["fgsm"]["acc_before"]:.1f}%</b> detection
              &nbsp;→&nbsp;
              <b style='color:{COLORS["cyan"]}'>After: </b>
              <b style='color:{COLORS["red"]}'>{summary["fgsm"]["acc_after"]:.1f}%</b> detection<br>
              <span class="badge badge-red">Evasion Rate: {summary["fgsm"]["evasion_rate"]:.1f}%</span>
              <span class="badge badge-orange">ε = 0.1</span>
              <span class="badge badge-cyan">1 step</span>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="card card-red">
              <h3>🔄 PGD Attack (Projected Gradient Descent)</h3>
              <p>Iterative attack — 40 small gradient steps, each projected
              back to stay within budget. The gold-standard adversarial attack.</p>
              <br>
              <b style='color:{COLORS["cyan"]}'>Before: </b>
              <b style='color:{COLORS["green"]}'>{summary["pgd"]["acc_before"]:.1f}%</b> detection
              &nbsp;→&nbsp;
              <b style='color:{COLORS["cyan"]}'>After: </b>
              <b style='color:{COLORS["red"]}'>{summary["pgd"]["acc_after"]:.1f}%</b> detection<br>
              <span class="badge badge-red">Evasion Rate: {summary["pgd"]["evasion_rate"]:.1f}%</span>
              <span class="badge badge-orange">ε = 0.1</span>
              <span class="badge badge-purple">40 steps</span>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        tab1, tab2, tab3 = st.tabs(["Attack Comparison", "Probability Shift", "Feature Perturbation"])

        with tab1:
            if os.path.exists("dashboard/assets/attack_comparison.png"):
                st.image("dashboard/assets/attack_comparison.png", use_container_width=True)

        with tab2:
            if models_loaded:
                n = min(80, len(X_adv_fgsm))
                p_clean = baseline_model.predict(X_orig_fraud[:n], verbose=0).flatten()
                p_fgsm  = baseline_model.predict(X_adv_fgsm[:n],  verbose=0).flatten()
                p_pgd   = baseline_model.predict(X_adv_pgd[:n],   verbose=0).flatten()

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=p_clean, mode="lines", name="Clean Input",
                    line=dict(color=COLORS["cyan"], width=2)
                ))
                fig.add_trace(go.Scatter(
                    y=p_fgsm, mode="lines", name="After FGSM",
                    line=dict(color=COLORS["orange"], width=2)
                ))
                fig.add_trace(go.Scatter(
                    y=p_pgd, mode="lines", name="After PGD",
                    line=dict(color=COLORS["red"], width=2.5, dash="dash")
                ))
                fig.add_hline(y=threshold, line_dash="dot", line_color=COLORS["purple"],
                              annotation_text=f"Decision threshold ({threshold:.3f})",
                              annotation_font_color=COLORS["purple"])
                fig.update_layout(
                    title="Fraud Probability Collapse Under Attack",
                    xaxis_title="Transaction Index",
                    yaxis_title="Predicted Fraud Probability",
                    paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["surface"],
                    font=dict(color=COLORS["text"]),
                    yaxis=dict(range=[0, 1.1], gridcolor=COLORS["border"]),
                    xaxis=dict(gridcolor=COLORS["border"]),
                    legend=dict(bgcolor=COLORS["surface"]),
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Each line shows the model's fraud probability for the same transactions. "
                           "Samples that drop below the threshold are classified as normal — the attack succeeded.")

        with tab3:
            col_img, col_info = st.columns([3, 1])
            with col_img:
                if os.path.exists("dashboard/assets/feature_perturbation.png"):
                    st.image("dashboard/assets/feature_perturbation.png", use_container_width=True)
            with col_info:
                st.markdown(f"""
                <div class="card">
                  <h3>What This Shows</h3>
                  <p>The bar height = how much each feature was changed on average.<br><br>
                  FGSM makes one large change.<br>
                  PGD distributes changes across features more precisely.<br><br>
                  <b style='color:{COLORS["cyan"]}'>All changes ≤ ε = 0.1</b> — invisible to humans.</p>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.info("Run the pipeline first: `python run_pipeline.py`")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DEFENSE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Defense":
    st.markdown('<div class="hero-title">Defense</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Two strategies to make the model robust against adversarial attacks</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="card card-green">
          <h3>🏋️ Defense 1 — Adversarial Training</h3>
          <p>During each training batch, we replace 50% of samples with
          FGSM-attacked versions. The model learns to classify correctly
          even under perturbation.</p>
          <br>
          <span class="badge badge-green">Training-time defense</span>
          <span class="badge badge-cyan">50% adversarial ratio</span>
          <span class="badge badge-orange">FGSM ε=0.05</span>
          <br><br>
          <p><b>Analogy:</b> Like training a security guard by having colleagues
          wear disguises — practice makes perfect.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card card-cyan">
          <h3>🧹 Defense 2 — Feature Squeezing</h3>
          <p>Reduce input precision by quantizing to N bits before prediction.
          Adversarial perturbations are tiny and precise — rounding destroys them.
          The main fraud signal (large statistical patterns) survives.</p>
          <br>
          <span class="badge badge-cyan">Inference-time defense</span>
          <span class="badge badge-purple">No retraining needed</span>
          <span class="badge badge-orange">4-bit default</span>
          <br><br>
          <p><b>Analogy:</b> Like glasses that blur tiny scratches but still
          let you see faces clearly.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    if os.path.exists("dashboard/assets/defense_comparison.png"):
        st.markdown('<div class="section-title">Vulnerable vs Defended Model</div>', unsafe_allow_html=True)
        col_img, col_analysis = st.columns([3, 2])
        with col_img:
            st.image("dashboard/assets/defense_comparison.png", use_container_width=True)
        with col_analysis:
            if data_loaded and summary:
                sq_f = summary["squeeze_fgsm"]
                sq_p = summary["squeeze_pgd"]
                st.markdown(f"""
                <div class="card" style='margin-bottom:.8rem;'>
                  <h3>Feature Squeezing Results</h3>
                  <p>
                  <b>FGSM:</b><br>
                  Adversarial recall: <b style='color:{COLORS["red"]}'>{sq_f["rec_adv"]*100:.1f}%</b><br>
                  After squeezing:   <b style='color:{COLORS["green"]}'>{sq_f["rec_squeezed"]*100:.1f}%</b>
                  &nbsp;<span class="badge badge-green">+{(sq_f["rec_squeezed"]-sq_f["rec_adv"])*100:.1f}%</span>
                  <br><br>
                  <b>PGD:</b><br>
                  Adversarial recall: <b style='color:{COLORS["red"]}'>{sq_p["rec_adv"]*100:.1f}%</b><br>
                  After squeezing:   <b style='color:{COLORS["green"]}'>{sq_p["rec_squeezed"]*100:.1f}%</b>
                  &nbsp;<span class="badge badge-green">+{(sq_p["rec_squeezed"]-sq_p["rec_adv"])*100:.1f}%</span>
                  </p>
                </div>
                """, unsafe_allow_html=True)

    # ── Interactive squeezing ─────────────────────────────────────────────────
    if data_loaded and models_loaded:
        st.divider()
        st.markdown('<div class="section-title">Interactive: Feature Squeezing Simulator</div>',
                    unsafe_allow_html=True)

        bit_depth = st.slider("Bit Depth (lower = more smoothing = stronger defense)", 2, 8, 4)

        from defense.adversarial_defense import feature_squeezing as fs
        n = min(100, len(X_adv_fgsm))
        X_sq = fs(X_adv_fgsm[:n], bit_depth=bit_depth)

        p_adv = baseline_model.predict(X_adv_fgsm[:n], verbose=0).flatten()
        p_sq  = baseline_model.predict(X_sq,           verbose=0).flatten()

        from sklearn.metrics import recall_score
        rec_adv = recall_score(y_fraud[:n], (p_adv >= threshold).astype(int), zero_division=0)
        rec_sq  = recall_score(y_fraud[:n], (p_sq  >= threshold).astype(int), zero_division=0)

        c1, c2, c3 = st.columns(3)
        c1.metric("Bit Depth", bit_depth, f"{2**bit_depth} precision levels")
        c2.metric("Recall (adversarial)", f"{rec_adv*100:.1f}%")
        c3.metric("Recall (after squeezing)", f"{rec_sq*100:.1f}%",
                  delta=f"{(rec_sq-rec_adv)*100:+.1f}%")

        # Scatter: adversarial prob vs squeezed prob
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=p_adv, y=p_sq, mode="markers",
            marker=dict(color=COLORS["cyan"], size=6, opacity=0.6),
            name="Transactions"
        ))
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(color=COLORS["muted"], dash="dash"))
        fig.add_vline(x=threshold, line_color=COLORS["red"],   line_dash="dot",
                      annotation_text="Adversarial threshold")
        fig.add_hline(y=threshold, line_color=COLORS["green"], line_dash="dot",
                      annotation_text="Squeezed threshold")
        fig.update_layout(
            title=f"Adversarial Probability vs Squeezed Probability (bit_depth={bit_depth})",
            xaxis_title="Prob. (adversarial input)",
            yaxis_title="Prob. (after squeezing)",
            paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["surface"],
            font=dict(color=COLORS["text"]),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Points above the diagonal = squeezing increased the fraud probability (defense worked). "
                   "Points right of threshold but below = adversarial succeeded. "
                   "Points right of threshold and above threshold line = recovered by squeezing.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE DEMO
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Live Demo":
    st.markdown('<div class="hero-title">Live Demo</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">See the attack happen in real-time on real fraud transactions</div>',
                unsafe_allow_html=True)

    if not models_loaded:
        st.error("Run `python run_pipeline.py` first to generate models.")
        st.stop()

    col_opt1, col_opt2 = st.columns([1, 2])
    with col_opt1:
        use_sample = st.button("Use Sample Fraud Transactions", type="primary",
                               use_container_width=True)
    with col_opt2:
        uploaded = st.file_uploader(
            "Or upload CSV (columns: V1-V28, Amount)", type=["csv"],
            label_visibility="collapsed"
        )

    X_demo = None

    if use_sample and data_loaded:
        X_demo = X_orig_fraud[:8]
        st.success(f"Loaded 8 real fraud transactions from test set.")

    elif uploaded:
        df = pd.read_csv(uploaded)
        for col in ["Class", "Time"]:
            if col in df.columns:
                df = df.drop(columns=[col])
        if "Amount" in df.columns:
            from sklearn.preprocessing import StandardScaler
            df["Amount"] = StandardScaler().fit_transform(df[["Amount"]])
        X_demo = df.values.astype(np.float32)
        st.success(f"Loaded {len(X_demo)} transactions.")

    if X_demo is not None:
        st.divider()

        from attacks.adversarial_attacks import wrap_model_for_art
        from art.attacks.evasion import FastGradientMethod

        classifier_live = wrap_model_for_art(baseline_model, input_shape=(X_demo.shape[1],))
        attack_live     = FastGradientMethod(estimator=classifier_live, eps=0.1)
        X_adv_live      = attack_live.generate(x=X_demo.astype(np.float32))

        p_orig = baseline_model.predict(X_demo,      verbose=0).flatten()
        p_adv  = baseline_model.predict(X_adv_live,  verbose=0).flatten()

        # ── Side-by-side results ──────────────────────────────────────────────
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(f'<div class="section-title">Original Transactions</div>', unsafe_allow_html=True)
            for i, p in enumerate(p_orig):
                is_fraud = p >= threshold
                color  = COLORS["red"] if is_fraud else COLORS["green"]
                label  = "FRAUD DETECTED" if is_fraud else "Normal"
                icon   = "🚨" if is_fraud else "✅"
                st.markdown(
                    f"<div style='background:{COLORS['surface']};border:1px solid {color};"
                    f"border-radius:8px;padding:.5rem 1rem;margin-bottom:.4rem;'>"
                    f"{icon} <b>Tx {i+1}</b> — "
                    f"<b style='color:{color}'>{label}</b> "
                    f"<span style='color:{COLORS['muted']};font-size:.85rem;'>({p*100:.1f}% fraud prob.)</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            fig_orig = go.Figure(go.Bar(
                x=[f"Tx {i+1}" for i in range(len(p_orig))],
                y=p_orig * 100,
                marker_color=[COLORS["red"] if p >= threshold else COLORS["green"] for p in p_orig],
                text=[f"{p*100:.1f}%" for p in p_orig],
                textposition="outside", textfont=dict(color="white"),
            ))
            fig_orig.add_hline(y=threshold*100, line_dash="dot",
                               line_color=COLORS["purple"], annotation_text="Threshold")
            fig_orig.update_layout(
                title="Fraud Probability — Original",
                yaxis_range=[0, 125], height=300,
                paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["surface"],
                font=dict(color=COLORS["text"]),
                yaxis=dict(gridcolor=COLORS["border"]),
                margin=dict(t=40, b=10),
            )
            st.plotly_chart(fig_orig, use_container_width=True)

        with col_b:
            st.markdown(f'<div class="section-title">After FGSM Attack (ε=0.1)</div>', unsafe_allow_html=True)
            for i, (po, pa) in enumerate(zip(p_orig, p_adv)):
                was_fraud  = po >= threshold
                now_fraud  = pa >= threshold
                if was_fraud and not now_fraud:
                    color, label, icon = COLORS["orange"], "EVADED — Attack Succeeded!", "⚠️"
                elif now_fraud:
                    color, label, icon = COLORS["red"], "Still Detected", "🛡️"
                else:
                    color, label, icon = COLORS["green"], "Normal (unchanged)", "✅"
                st.markdown(
                    f"<div style='background:{COLORS['surface']};border:1px solid {color};"
                    f"border-radius:8px;padding:.5rem 1rem;margin-bottom:.4rem;'>"
                    f"{icon} <b>Tx {i+1}</b> — "
                    f"<b style='color:{color}'>{label}</b> "
                    f"<span style='color:{COLORS['muted']};font-size:.85rem;'>({pa*100:.1f}% fraud prob.)</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            fig_adv = go.Figure(go.Bar(
                x=[f"Tx {i+1}" for i in range(len(p_adv))],
                y=p_adv * 100,
                marker_color=[COLORS["orange"] if (po >= threshold and pa < threshold)
                              else (COLORS["red"] if pa >= threshold else COLORS["green"])
                              for po, pa in zip(p_orig, p_adv)],
                text=[f"{p*100:.1f}%" for p in p_adv],
                textposition="outside", textfont=dict(color="white"),
            ))
            fig_adv.add_hline(y=threshold*100, line_dash="dot",
                              line_color=COLORS["purple"], annotation_text="Threshold")
            fig_adv.update_layout(
                title="Fraud Probability — After Attack",
                yaxis_range=[0, 125], height=300,
                paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["surface"],
                font=dict(color=COLORS["text"]),
                yaxis=dict(gridcolor=COLORS["border"]),
                margin=dict(t=40, b=10),
            )
            st.plotly_chart(fig_adv, use_container_width=True)

        # ── Feature perturbation ──────────────────────────────────────────────
        st.divider()
        st.markdown('<div class="section-title">Feature Perturbation Viewer</div>', unsafe_allow_html=True)

        tx_idx = st.selectbox("Select transaction", [f"Tx {i+1}" for i in range(len(X_demo))])
        idx    = int(tx_idx.split()[1]) - 1
        diff   = np.abs(X_adv_live[idx] - X_demo[idx])

        fig_diff = px.bar(
            x=[f"V{i+1}" for i in range(min(28, len(diff)))],
            y=diff[:28],
            labels={"x": "Feature", "y": "Absolute Change"},
            title=f"Feature Changes in {tx_idx} After FGSM Attack",
            color=diff[:28],
            color_continuous_scale=[[0, COLORS["surface"]], [0.5, COLORS["orange"]], [1, COLORS["red"]]],
        )
        fig_diff.update_layout(
            paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["surface"],
            font=dict(color=COLORS["text"]),
            coloraxis_showscale=False,
            height=320, margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_diff, use_container_width=True)
        st.caption(f"Max change: {diff.max():.6f} — Min change: {diff.min():.6f} — All within ε=0.1 budget")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "How It Works":
    st.markdown('<div class="hero-title">How It Works</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Deep dives into the math, intuition, and Q&A</div>',
                unsafe_allow_html=True)

    with st.expander("⚡ FGSM — Fast Gradient Sign Method", expanded=True):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"""
            **Invented by:** Ian Goodfellow et al., Google Brain (2014)

            **The Formula:**
            """)
            st.markdown(f"""
            <div class="formula-box">
x_adv = x + ε × sign( ∇ₓ Loss(model, x, y_true) )
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            **Step by step:**
            1. Feed fraud transaction `x` through the model
            2. Compute gradient of loss w.r.t. input (not weights!)
            3. Take the sign: +1 if gradient is positive, -1 if negative
            4. Multiply by ε (perturbation budget)
            5. Add to original — one step in the worst direction
            """)
        with col2:
            st.markdown(f"""
            <div class="card card-orange">
              <h3>Intuition</h3>
              <p>The gradient tells us: "which direction in feature space increases
              the model's loss (mistake) the most?"<br><br>
              FGSM takes <b>one large step</b> in that direction.
              Like nudging a ball sitting on a hill — one push, and it rolls down
              to the wrong side.</p>
            </div>
            <div class="card">
              <h3>Parameters</h3>
              <p>
              <b>ε = 0.1</b> — max change per feature<br>
              <b>1 step</b> — single gradient computation<br>
              <b>Speed</b> — very fast (1 forward + 1 backward pass)
              </p>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("🔄 PGD — Projected Gradient Descent"):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Invented by:** Madry et al., MIT (2017)")
            st.markdown(f"""
            <div class="formula-box">
x⁽⁰⁾   = x
x⁽ᵗ⁺¹⁾ = Π_ε( x⁽ᵗ⁾ + α × sign(∇ₓ Loss) )

Π_ε = project back into ε-ball if exceeded
α   = 0.01  (step size per iteration)
T   = 40    (iterations)
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            **Why stronger than FGSM:**
            - FGSM takes one big step → may overshoot
            - PGD takes 40 small precise steps → finds optimal perturbation
            - Each step is projected back so total change ≤ ε
            - Considered the **gold standard** white-box attack
            """)
        with col2:
            st.markdown(f"""
            <div class="card card-red">
              <h3>Intuition</h3>
              <p>Instead of one big jump across the decision boundary,
              PGD takes 40 tiny careful steps — each time checking it hasn't
              gone too far (projection step).<br><br>
              Like picking a lock: FGSM tries to kick the door down.
              PGD carefully manipulates each pin one by one.</p>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("🛡️ Why Adversarial Training Works"):
        st.markdown(f"""
        <div class="card card-green">
          <h3>The Core Idea</h3>
          <p>The model's weakness is that it has never seen adversarial examples during training.
          Adversarial training fixes this by including them in every batch.<br><br>
          The model learns: <i>"Even when these features are slightly shifted in the adversarial direction,
          this is still fraud."</i> The decision boundary moves to cover the adversarial region.</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🎓 Viva / Exam Q&A"):
        qa_pairs = [
            ("What is an adversarial attack?",
             "A carefully crafted input perturbation that causes a model to misclassify, while the change is imperceptible to humans. Formally: finding δ such that ||δ||∞ ≤ ε but f(x+δ) ≠ f(x)."),
            ("Why is PR-AUC better than ROC-AUC for fraud detection?",
             "On severely imbalanced datasets, ROC-AUC is overly optimistic because it uses True Negative Rate which is inflated by the majority class. PR-AUC focuses on the minority class (fraud) and is more informative when positives are rare."),
            ("What is the difference between FGSM and PGD?",
             "FGSM takes one step of size ε in the gradient direction. PGD takes T small steps of size α, projecting back to the ε-ball after each step. PGD is strictly stronger — it finds the optimal perturbation within the budget."),
            ("Why use Focal Loss for fraud detection?",
             "Standard cross-entropy treats all examples equally. Focal Loss down-weights easy negatives (transactions the model already classifies correctly with high confidence) and focuses learning on hard cases — the borderline fraud transactions that matter most."),
            ("What is the evasion rate metric?",
             "The percentage of fraud transactions that the attack successfully made the model classify as normal. Calculated as: (fraud correctly detected before attack, now misclassified) / (fraud correctly detected before attack) × 100."),
            ("What are the limitations of feature squeezing?",
             "1) May reduce clean accuracy slightly. 2) A sophisticated attacker can craft adversarial examples that survive quantization (adaptive attacks). 3) Optimal bit depth requires tuning per dataset."),
            ("What is a white-box attack?",
             "An attack where the adversary has full knowledge of the model: architecture, weights, and training data. This is the worst-case scenario. Our experiments use white-box attacks to measure the theoretical maximum vulnerability."),
            ("Why do we use SMOTE only on training data?",
             "SMOTE generates synthetic samples by interpolating between real fraud examples. Applying it to the test set would give an unrealistically easy evaluation. The test set must reflect real-world class distribution to give honest performance estimates."),
        ]

        for q, a in qa_pairs:
            st.markdown(f"""
            <div style='background:{COLORS["surface"]};border:1px solid {COLORS["border"]};
                        border-radius:10px;padding:1rem 1.2rem;margin-bottom:.6rem;'>
              <b style='color:{COLORS["cyan"]};'>Q: {q}</b><br>
              <span style='color:{COLORS["text"]};font-size:.9rem;'>✅ {a}</span>
            </div>
            """, unsafe_allow_html=True)
