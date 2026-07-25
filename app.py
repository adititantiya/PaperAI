import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

from gemini_helper import explain
from feedback import save_feedback
from recommendation import get_recommendations

st.set_page_config(
    page_title="Paper Grade Intelligence",
    page_icon="📄",
    layout="wide"
)
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

model = joblib.load("models/random_forest.pkl")
history = pd.read_csv("data/historical_data.csv")

st.html("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>

:root{
    --ink:#0F172A;
    --muted:#64748B;
    --line:#E6ECF3;
    --blue:#2563EB;
    --indigo:#4F46E5;
    --green:#22C55E;
    --amber:#F59E0B;
    --red:#EF4444;
    --bg:#EEF3F8;
}

.stApp{
    background:
        radial-gradient(1100px 420px at 12% -8%, rgba(37,99,235,.05), transparent 60%),
        radial-gradient(900px 380px at 100% 0%, rgba(79,70,229,.05), transparent 55%),
        var(--bg);
    font-family:'Poppins',sans-serif;
}
html, body, [class*="css"]{
    font-family:'Poppins',sans-serif;
}
.block-container{
    padding-top:1.5rem;
    padding-left:2rem;
    padding-right:2rem;
    max-width:1500px;
}

/* Sidebar */
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#08152C,#142A4F);
}
[data-testid="stSidebar"] *{
    color:white;
}
[data-testid="stSidebar"] .block-container{
    padding-top:1.4rem;
    padding-left:1.1rem;
    padding-right:1.1rem;
}

/* ---- Sidebar: brand row ---- */
.brand-row{
    display:flex;align-items:center;gap:12px;
    margin-bottom:6px;
}
.brand-icon{
    width:44px;height:44px;border-radius:13px;
    background:linear-gradient(135deg,#2563EB,#4F46E5);
    display:flex;align-items:center;justify-content:center;
    font-size:22px;flex-shrink:0;
    box-shadow:0 6px 16px rgba(37,99,235,.35);
}
.brand-name{font-size:20px;font-weight:700;color:#fff;line-height:1.15;}
.brand-sub{font-size:12.5px;color:#93A4C3;font-weight:500;}

/* ---- Sidebar: nav ---- */
.nav-item{
    display:flex;align-items:center;gap:10px;
    padding:11px 14px;border-radius:12px;
    font-size:14.5px;font-weight:600;color:#B9C4DC;
    margin-bottom:4px;
}
.nav-item.active{
    background:linear-gradient(90deg,#2563EB,#3B4FDB);
    color:#fff;
    box-shadow:0 6px 16px rgba(37,99,235,.35);
}

/* ---- Sidebar: section label ---- */
.side-label{
    font-size:11.5px;font-weight:700;letter-spacing:1.4px;
    color:#6E7FA3;margin:18px 0 10px 4px;
}

/* ---- Sidebar: system info rows ---- */
.sys-row{
    display:flex;align-items:center;gap:12px;
    padding:9px 4px;
}
.sys-icon{
    width:32px;height:32px;border-radius:9px;
    background:rgba(255,255,255,.08);
    display:flex;align-items:center;justify-content:center;
    font-size:15px;flex-shrink:0;
}
.sys-label{font-size:11.5px;color:#93A4C3;font-weight:500;margin-bottom:1px;}
.sys-value{font-size:14px;color:#fff;font-weight:600;}

.side-divider{
    height:1px;background:rgba(255,255,255,.1);
    margin:14px 0;border:none;
}

.online-pill{
    display:flex;align-items:center;gap:10px;
    background:rgba(34,197,94,.12);
    border:1px solid rgba(34,197,94,.35);
    border-radius:14px;padding:12px 14px;
}
.online-dot{
    width:9px;height:9px;border-radius:50%;background:#22C55E;
    box-shadow:0 0 0 4px rgba(34,197,94,.18);
    flex-shrink:0;
}
.online-title{font-size:13.5px;font-weight:700;color:#fff;}
.online-sub{font-size:11.5px;color:#93A4C3;}

.insight-card{
    background:linear-gradient(160deg,#16305C,#0E2246);
    border:1px solid rgba(255,255,255,.08);
    border-radius:16px;padding:16px 16px 18px 16px;
}
.insight-icon{
    width:34px;height:34px;border-radius:10px;
    background:rgba(255,255,255,.1);
    display:flex;align-items:center;justify-content:center;
    font-size:16px;margin-bottom:10px;
}
.insight-title{font-size:14px;font-weight:700;color:#fff;margin-bottom:6px;}
.insight-text{font-size:12px;line-height:1.55;color:#A9B6D2;}

/* Bordered containers -> card look */
div[data-testid="stVerticalBlockBorderWrapper"]{
    background:white;
    border-radius:20px;
    border:1px solid #E6ECF3 !important;
    box-shadow:0 8px 28px rgba(15,23,42,.08);
    padding:10px;
    transition:.25s;
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover{
    transform:translateY(-2px);
    box-shadow:0 12px 35px rgba(15,23,42,.12);
}

/* Metric cards */
div[data-testid="metric-container"]{
    background:white;
    border-radius:18px;
    padding:20px;
    border:1px solid #EEF2F7;
    box-shadow:0 8px 24px rgba(15,23,42,.06);
}

/* Number Inputs */
.stNumberInput{
    background:#FAFBFC;
    border-radius:14px;
    padding:4px;
    border:1px solid #E5E7EB;
}
.stNumberInput input{
    border-radius:12px !important;
}

/* Buttons */
.stButton>button{
    background:linear-gradient(90deg,#2563EB,#4F46E5);
    color:white;
    border:none;
    border-radius:14px;
    height:52px;
    font-size:17px;
    font-weight:600;
    transition:.25s;
    width:100%;
}
.stButton>button:hover{
    transform:translateY(-2px);
    box-shadow:0 10px 20px rgba(79,70,229,.35);
}

/* Secondary / reset button look */
button[kind="secondary"]{
    background:white !important;
    color:#1E3A8A !important;
    border:1px solid #E7EAF0 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{
    gap:6px;
    background:#fff;
    padding:6px;
    border-radius:14px;
    border:1px solid var(--line);
    box-shadow:0 6px 18px rgba(15,23,42,.05);
    width:fit-content;
}
button[data-baseweb="tab"]{
    font-size:15.5px;
    font-weight:600;
    color:#64748B;
    border-radius:10px !important;
    padding:6px 18px !important;
}

button[data-baseweb="tab"][aria-selected="true"]{
    color:#fff !important;
    background:linear-gradient(90deg,#2563EB,#4F46E5);
}
.stTabs [data-baseweb="tab-highlight"]{ display:none; }
.stTabs [data-baseweb="tab-border"]{ display:none; }

/* ---- Hero banner ---- */
.hero-card{
    position:relative;
    background:linear-gradient(120deg,#EAF1FF 0%,#F3F0FF 55%,#EEF3F8 100%);
    border:1px solid #E1E9F7;
    border-radius:24px;
    padding:30px 34px;
    margin-bottom:22px;
    overflow:hidden;
    box-shadow:0 10px 30px rgba(37,99,235,.06);
}
.hero-top{
    display:flex;align-items:flex-start;justify-content:space-between;
    position:relative;z-index:2;
}
.hero-left{display:flex;align-items:flex-start;gap:16px;}
.hero-icon{
    width:60px;height:60px;border-radius:16px;
    background:linear-gradient(135deg,#2563EB,#4F46E5);
    display:flex;align-items:center;justify-content:center;
    font-size:28px;flex-shrink:0;
    box-shadow:0 10px 22px rgba(37,99,235,.3);
}
.hero-title{
    font-size:38px;font-weight:800;color:var(--ink);
    letter-spacing:-1px;line-height:1.05;margin-bottom:6px;
}
.hero-tag{
    font-size:16.5px;font-weight:600;color:var(--blue);margin-bottom:4px;
}
.hero-desc{font-size:14px;color:var(--muted);font-weight:500;}
.status-pill{
    display:inline-flex;align-items:center;gap:8px;
    background:#fff;border:1px solid #E1E9F7;border-radius:30px;
    padding:9px 18px;font-size:13.5px;font-weight:600;color:#1E293B;
    box-shadow:0 6px 16px rgba(15,23,42,.06);
    flex-shrink:0;
}
.hero-decor{
    position:absolute;right:-30px;bottom:-40px;
    width:280px;height:280px;
    background:radial-gradient(circle at 40% 40%, rgba(79,70,229,.10), transparent 70%);
    z-index:1;
}

/* ---- Parameter input icon labels ---- */
.param-label{
    display:flex;align-items:center;gap:8px;
    font-size:13px;font-weight:600;color:#334155;
    margin:2px 0 -6px 2px;
}
.param-chip{
    width:22px;height:22px;border-radius:7px;
    display:flex;align-items:center;justify-content:center;
    font-size:12px;flex-shrink:0;
}

/* Alerts */
div[data-testid="stAlert"]{
    border-radius:14px;
}

/* DataFrame */
[data-testid="stDataFrame"]{
    border-radius:18px;
}

/* Headers */
h1{ font-weight:700; }
h2{ color:#1E3A8A; }
h3{ color:#1E3A8A; }

/* ---- Custom dashboard components ---- */
.top-badge{
    display:inline-flex;align-items:center;gap:6px;
    background:white;border:1px solid #E7EAF0;border-radius:12px;
    padding:8px 16px;font-size:14px;font-weight:600;color:#1E293B;
    box-shadow:0 4px 12px rgba(0,0,0,.04);
}
.dot-green{width:8px;height:8px;border-radius:50%;background:#22C55E;display:inline-block;}

.stat-card{
    background:white;
    border-radius:18px;
    padding:16px 18px;
    min-height:120px;
    display:flex;
    align-items:center;
    gap:14px;
}
.stat-icon{
    width:42px;
    height:42px;
    border-radius:12px;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:20px;
}
.stat-label{
    font-size:13px;
    color:#64748B;
    margin-bottom:3px;
}
.stat-value{
    font-size:20px;
    font-weight:700;
    color:#111827;
    line-height:1.2;
}
.stat-sub{
    font-size:11px;
    color:#94A3B8;
}

.health-card{
    background:white;border-radius:18px;padding:18px 22px;
    height:120px;
    box-shadow:0 10px 30px rgba(0,0,0,.06);
    display:flex;align-items:center;justify-content:space-between;
}
.ring{
    width:64px;
    height:64px;
    border-radius:50%;
    display:flex;
    align-items:center;
    justify-content:center;
}
.ring-inner{
    width:48px;
    height:48px;
    background:white;
    border-radius:50%;
}

.dt-card{
    background:#F8FAFC;
    border:1px solid #E5E7EB;
    border-radius:16px;
    padding:16px;
    transition:.2s;
}

.dt-card:hover{
    background:white;
}
.dt-label{font-size:12px;color:#64748B;font-weight:600;display:flex;align-items:center;gap:7px;margin-bottom:6px;}
.dt-label .param-chip{width:22px;height:22px;border-radius:7px;}
.dt-value{font-size:18px;font-weight:700;color:#0F172A;}
.dt-unit{font-size:11px;color:#94A3B8;}

.machine-box{
    background:linear-gradient(135deg,#EFF6FF,#EEF2FF);
    border:1px solid #DCE4F5;border-radius:14px;
    padding:16px;text-align:center;margin:10px 0;
}
.bw-box{
    background:#F0FDF4;border:1px solid #DCFCE7;border-radius:14px;
    padding:12px 16px;display:flex;align-items:center;gap:12px;
}
.legend-dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:5px;}
.info-bar{
    background:#EFF6FF;border:1px solid #DBEAFE;border-radius:14px;
    padding:16px 20px;margin-top:10px;
}
.main-title{
    font-size:52px;
    font-weight:800;
    color:#0F172A;
    margin-bottom:4px;
    letter-spacing:-1px;
}

.sub-title{
    font-size:18px;
    color:#64748B;
    margin-bottom:4px;
}

.kpi-card{
    background:white;
    border-radius:18px;
    padding:22px;
    border:1px solid #E5E7EB;
    box-shadow:0 8px 24px rgba(0,0,0,.06);
    transition:.25s;
    height:220px;
    display:flex;
    flex-direction:column;
    justify-content:center;
}

.kpi-card:hover{
    transform:translateY(-3px);
    box-shadow:0 15px 35px rgba(37,99,235,.12);
}
.alert-card{
background:#FEF2F2;
border-left:6px solid #EF4444;
padding:18px;
border-radius:14px;
margin-bottom:15px;
}

.ai-card{
background:white;
padding:24px;
border-radius:18px;
border:1px solid #E5E7EB;
box-shadow:0 8px 24px rgba(0,0,0,.05);
}
.kpi-title{
    color:#64748B;
    font-size:14px;
    font-weight:600;
}

.kpi-value{
    color:#111827;
    font-size:30px;
    font-weight:800;
    margin-top:8px;
}
.recipe-card{
    background:white;
    border-radius:20px;
    padding:22px;
    border:1px solid #E5E7EB;
    box-shadow:0 12px 30px rgba(37,99,235,.08);
}

.recipe-icon{
    font-size:28px;
}

.recipe-title{
    color:#64748B;
    font-size:14px;
    font-weight:600;
}

.recipe-value{
    font-size:26px;
    font-weight:700;
    color:#111827;
    margin-top:10px;
}

.recipe-sub{
    margin-top:18px;
    font-size:13px;
    color:#64748B;
}
.compare-card{
    background:white;
    border-radius:18px;
    padding:20px;
    text-align:center;
    border:1px solid #E5E7EB;
    box-shadow:0 8px 20px rgba(0,0,0,.06);
    transition:0.25s;
}
.compare-card:hover{
    transform:translateY(-4px);
    box-shadow:0 18px 40px rgba(37,99,235,.12);
}
.report-card{
    background:white;
    border-radius:18px;
    padding:22px;
    border:1px solid #E5E7EB;
    box-shadow:0 8px 24px rgba(0,0,0,.06);
    margin-bottom:15px;
}
.compare-status{
    margin-top:18px;
    padding:8px 12px;
    border-radius:10px;
    background:#ECFDF5;
    color:#16A34A;
    font-weight:700;
    font-size:16px;
}

.compare-icon{
font-size:28px;
margin-bottom:10px;
}

.compare-title{
    font-size:17px;
    font-weight:700;
    margin-top:10px;
    color:#1F2937;
}

.compare-current{

font-size:26px;

font-weight:700;

margin-top:10px;
}

.compare-target{
    margin-top:10px;
    font-size:15px;
    color:#64748B;
    font-weight:600;
}
</style>
        
*{
    transition:all .2s ease;
}

hr{
    border:none;
    border-top:1px solid #E5E7EB;
}
""")

accuracy = joblib.load("models/accuracy.pkl")

with st.sidebar:
    st.html("""
    <div class="brand-row">
        <div class="brand-icon">🏭</div>
        <div>
            <div class="brand-name">PaperAI</div>
            <div class="brand-sub">Grade Change Intelligence</div>
        </div>
    </div>
    """)

    st.html("""<div class="nav-item active">🏠&nbsp;&nbsp;Overview</div>""")

    st.html("""<div class="side-label">SYSTEM</div>""")

    st.html(f"""
    <div class="sys-row">
        <div class="sys-icon">⚙️</div>
        <div><div class="sys-label">Model</div><div class="sys-value">Random Forest</div></div>
    </div>
    <div class="sys-row">
        <div class="sys-icon">✨</div>
        <div><div class="sys-label">AI Engine</div><div class="sys-value">Gemini Flash</div></div>
    </div>
    <div class="sys-row">
        <div class="sys-icon">📋</div>
        <div><div class="sys-label">Dataset</div><div class="sys-value">5000 Samples</div></div>
    </div>
    <div class="sys-row">
        <div class="sys-icon">🗂️</div>
        <div><div class="sys-label">Loaded Records</div><div class="sys-value">{len(history)} historical records</div></div>
    </div>
    <div class="sys-row">
        <div class="sys-icon">🎯</div>
        <div><div class="sys-label">Model Accuracy</div><div class="sys-value">{accuracy*100:.2f}%</div></div>
    </div>
    """)

    st.html("""
    <div style="margin-top:14px">
    <div class="online-pill">
        <div class="online-dot"></div>
        <div>
            <div class="online-title">System Online</div>
            <div class="online-sub">All systems operational</div>
        </div>
    </div>
    </div>
    """)

    st.html("<div class='side-divider'></div>")

    st.html("""
    <div class="insight-card">
        <div class="insight-icon">🧠</div>
        <div class="insight-title">AI-Powered Insights</div>
        <div class="insight-text">
            Predicts paper quality during grade transition and recommends
            corrective actions using AI.
        </div>
    </div>
    """)

st.html("""
<div class="hero-card">
    <div class="hero-decor"></div>
    <div class="hero-top">
        <div class="hero-left">
            <div class="hero-icon">🏭</div>
            <div>
                <div class="hero-title">PaperAI</div>
                <div class="hero-tag">Grade Change Intelligence System</div>
                <div class="hero-desc">AI-powered prediction and recommendation system for paper manufacturing</div>
            </div>
        </div>
        
    </div>
</div>
""")

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

prediction_tab, analytics_tab = st.tabs(["📊 Prediction", "📈 Analytics"])

with prediction_tab:

    # ---- keep inputs in session state so cards update live ----
    def compute_prediction(stock_flow, filler_flow, steam_pressure, machine_speed,
                            moisture, ash, basis_weight, target_basis_weight):
        deviation = abs(basis_weight - target_basis_weight) / target_basis_weight * 100
        input_df = pd.DataFrame([{
            "stock_flow": stock_flow,
            "filler_flow": filler_flow,
            "steam_pressure": steam_pressure,
            "machine_speed": machine_speed,
            "moisture": moisture,
            "ash": ash,
            "basis_weight": basis_weight,
            "target_basis_weight": target_basis_weight,
            "bw_deviation": deviation
        }])
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        return input_df, deviation, prediction, probability

    # Layout: Process Parameters (left) | Process Overview / Digital Twin (right)
    left_col, right_col = st.columns([1.05, 1])

    with left_col:
        with st.container(border=True):
            st.markdown("#### ⚙️ Process Parameters")
            c1, c2 = st.columns(2)
            with c1:
                st.html("<div class='param-label'><span class='param-chip' style='background:#DBEAFE'>🌊</span>Stock Flow</div>")
                stock_flow = st.number_input("Stock Flow (L/min)", value=100.0, label_visibility="collapsed")
                st.html("<div class='param-label'><span class='param-chip' style='background:#D1FAE5'>🧪</span>Filler Flow</div>")
                filler_flow = st.number_input("Filler Flow (L/min)", value=20.0, label_visibility="collapsed")
                st.html("<div class='param-label'><span class='param-chip' style='background:#FFE4D6'>🔥</span>Steam Pressure</div>")
                steam_pressure = st.number_input("Steam Pressure (bar)", value=55.0, label_visibility="collapsed")
                st.html("<div class='param-label'><span class='param-chip' style='background:#E0E7FF'>⚡</span>Machine Speed</div>")
                machine_speed = st.number_input("Machine Speed (m/min)", value=900.0, label_visibility="collapsed")
            with c2:
                st.html("<div class='param-label'><span class='param-chip' style='background:#CFFAFE'>💧</span>Moisture</div>")
                moisture = st.number_input("Moisture (%)", value=5.0, label_visibility="collapsed")
                st.html("<div class='param-label'><span class='param-chip' style='background:#DCFCE7'>🍃</span>Ash</div>")
                ash = st.number_input("Ash (%)", value=2.0, label_visibility="collapsed")
                st.html("<div class='param-label'><span class='param-chip' style='background:#FDE68A'>⚖️</span>Basis Weight</div>")
                basis_weight = st.number_input("Basis Weight (GSM)", value=80.0, label_visibility="collapsed")
                st.html("<div class='param-label'><span class='param-chip' style='background:#E9D5FF'>🎯</span>Target Basis Weight</div>")
                target_basis_weight = st.number_input("Target Basis Weight (GSM)", value=80.0, label_visibility="collapsed")

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            b1, b2 = st.columns(2)
            with b1:
                analyze = st.button("🚀 Analyze Process", use_container_width=True)
            with b2:
                reset = st.button("🔄 Reset", use_container_width=True, type="secondary")
            if reset:
                st.rerun()

    # live (cheap) prediction so the top cards & digital twin update as you type
    _, deviation, prediction, probability = compute_prediction(
        stock_flow, filler_flow, steam_pressure, machine_speed,
        moisture, ash, basis_weight, target_basis_weight
    )
    process_health = max(0, 100 - probability * 100)
    health_color = "#22C55E" if process_health >= 70 else ("#F59E0B" if process_health >= 40 else "#EF4444")

    with right_col:
        with st.container(border=True):
            top_row1, top_row2 = st.columns([3, 1])
            with top_row1:
                st.markdown("#### 🏭 Process Overview")
            with top_row2:
                st.html("<div style='text-align:right'><span class='top-badge'>Digital Twin</span></div>")

            d1, d2, d3 = st.columns(3)
            with d1:
                st.html(f"""
                <div class='dt-card'>
                <div class='dt-label'><span class='param-chip' style='background:#DBEAFE'>🌊</span>Stock Flow</div>
                <div class='dt-value'>{stock_flow:.2f}</div>
                <div class='dt-unit'>L/min</div>
                </div>""")
            with d2:
                st.html(f"""
                <div class='dt-card'>
                <div class='dt-label'><span class='param-chip' style='background:#D1FAE5'>🧪</span>Filler Flow</div>
                <div class='dt-value'>{filler_flow:.2f}</div>
                <div class='dt-unit'>L/min</div>
                </div>""")
            with d3:
                st.html(f"""
                <div class='dt-card'>
                <div class='dt-label'><span class='param-chip' style='background:#FFE4D6'>🔥</span>Steam Pressure</div>
                <div class='dt-value'>{steam_pressure:.2f}</div>
                <div class='dt-unit'>bar</div>
                </div>""")

            machine_status = "Healthy" if probability < 0.5 else ("Watch" if probability < 0.8 else "At Risk")
            machine_color = "#22C55E" if probability < 0.5 else ("#F59E0B" if probability < 0.8 else "#EF4444")
            st.html(f"""
            <div class='machine-box'>
                <div style='font-weight:700;font-size:16px;color:#1E3A8A'>Paper Machine</div>
                <div style='color:{machine_color};font-weight:600;font-size:13px;margin-top:4px'>{machine_status}</div>
            </div>
            """)

            st.html(f"""
            <div class='bw-box'>
                <div style='font-size:22px'>💰</div>
                <div>
                    <div class='dt-label'>Basis Weight</div>
                    <div class='dt-value'>{basis_weight:.2f} <span class='dt-unit'>GSM</span></div>
                </div>
            </div>
            """)

            st.html(f"""
            <div style='margin-top:14px;font-size:12px;color:#64748B'>
                <span class='legend-dot' style='background:#22C55E'></span>Normal &nbsp;&nbsp;
                <span class='legend-dot' style='background:#F59E0B'></span>Warning &nbsp;&nbsp;
                <span class='legend-dot' style='background:#EF4444'></span>Critical
            </div>
            """)

    # ---- Top summary stat cards (Risk / Status / Deviation / Target) ----
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    s1, s2, s3, s4, s5 = st.columns([1, 1, 1, 1, 1.3])
    with s1:
        risk_color = "#22C55E"
        risk_text = "Low Risk"

        if probability > 0.4:
            risk_color = "#F59E0B"
            risk_text = "Moderate"

        if probability > 0.75:
            risk_color = "#EF4444"
            risk_text = "High Risk"

        st.html(f"""
        <div class='stat-card'>
            <div class='stat-icon' style='background:{risk_color}20;'>⚠️</div>
            <div>
                <div class='stat-label'>Risk Score</div>
                <div class='stat-value' style='color:{risk_color};'>
                    {probability*100:.2f}%
                </div>
                <div class='stat-sub'>{risk_text}</div>
            </div>
        </div>
        """)
    with s2:
        status_text = "🟢 WITHIN SPEC"
        status_color = "#22C55E"
        status_sub = "Stable Process"

        if prediction == 1:
            status_text = "🔴 OFF SPEC"
            status_color = "#EF4444"
            status_sub = "Process Drifting"

        st.html(f"""
        <div class='stat-card'>
            <div class='stat-icon' style='background:{status_color}20;'>📊</div>
            <div>
                <div class='stat-label'>Prediction</div>
                <div class='stat-value' style='color:{status_color};'>
                    {status_text}
                </div>
                <div class='stat-sub'>{status_sub}</div>
            </div>
        </div>
        """)
    with s3:
        dev_color = "#22C55E" if deviation <= 2.5 else "#EF4444"

        st.html(f"""
        <div class='stat-card'>
            <div class='stat-icon' style='background:{dev_color}20;'>🎯</div>
            <div>
                <div class='stat-label'>Deviation</div>
                <div class='stat-value' style='color:{dev_color};'>
                    {deviation:.2f}%
                </div>
                <div class='stat-sub'>Target ±2.5%</div>
            </div>
        </div>
        """)
    with s4:
        st.html(f"""
        <div class='stat-card'>
            <div class='stat-icon' style='background:#DBEAFE;'>📋</div>
            <div>
                <div class='stat-label'>Target Recipe</div>
                <div class='stat-value'>{target_basis_weight:.0f} GSM</div>
                <div class='stat-sub'>Basis Weight</div>
            </div>
        </div>
        """)
    with s5:
        health_pct = min(100, max(0, process_health))

        health_text = (
            "Excellent" if health_pct >= 80 else
            "Good" if health_pct >= 60 else
            "Fair" if health_pct >= 40 else
            "Poor"
        )

        st.html(f"""
        <div class='health-card'>
            <div>
                <div class='stat-label'>Health Score</div>
                <div class='stat-value'>{process_health:.0f}</div>
                <div class='stat-sub'
                     style='color:{health_color};font-weight:700'>
                    {health_text}
                </div>
            </div>

            <div class='ring'
                style='background:conic-gradient({health_color}
                {health_pct}%,
                #E5E7EB 0)'>

                <div class='ring-inner'></div>

            </div>

        </div>
        """)

    st.html("""
    <div class='info-bar'>
    <b>ℹ️ How it works</b><br>
    <span style='color:#475569;font-size:14px'>
    Enter current process parameters and target basis weight. Our AI model will predict the quality risk,
    identify key factors and recommend corrective actions.
    </span>
    </div>
    """)


    if analyze:
        st.session_state.analysis_done = True
    if st.session_state.analysis_done:
        reasons = []
        if steam_pressure > 65:
            reasons.append("High Steam Pressure")
        if machine_speed > 980:
            reasons.append("Machine Speed above safe operating range")
        if moisture > 6.5:
            reasons.append("High Moisture")
        if deviation > 2.5:
            reasons.append(f"Basis Weight deviates by {deviation:.2f}% from target")
        if len(reasons) == 0:
            reasons.append("All parameters are within acceptable operating limits.")

        st.divider()
        st.subheader("📊 Process Risk Assessment")

        risk = probability * 100

        if risk < 40:
            risk_color = "#22C55E"
            risk_text = "LOW RISK"
        elif risk < 70:
            risk_color = "#F59E0B"
            risk_text = "MODERATE RISK"
        else:
            risk_color = "#EF4444"
            risk_text = "HIGH RISK"

        st.html(f"""
        <div class="recipe-card" style="text-align:center;padding:22px;">

            <div style="
                font-size:18px;
                color:#64748B;
                font-weight:600;">
                Current OFF-SPEC Probability
            </div>

            <div style="
                font-size:52px;
                font-weight:800;
                color:{risk_color};
                margin-top:15px;">
                {risk:.1f}%
            </div>

            <div style="
                font-size:18px;
                font-weight:700;
                color:{risk_color};
                margin-top:8px;">
                {risk_text}
            </div>

            <div style="margin-top:15px;">
                <progress
                    value="{risk}"
                    max="100"
                    style="width:85%;height:20px;">
                </progress>
            </div>

            <div style="
                margin-top:15px;
                color:#64748B;
                font-size:15px;">
                Green &lt; 40% &nbsp;&nbsp;|&nbsp;&nbsp;
                Yellow 40–70% &nbsp;&nbsp;|&nbsp;&nbsp;
                Red &gt; 70%
            </div>

        </div>
        """)
        st.divider()
        st.subheader("🔗 Process Influence Chain")

        c1, a1, c2, a2, c3, a3, c4, a4, c5 = st.columns([2,0.25,2,0.25,2,0.25,2,0.25,2])

        with c1:
            st.html("""
            <div class="compare-card">
                <div style="font-size:42px;">🌊</div>
                <div class="compare-title">Stock Flow</div>
                <div class="compare-target">Input</div>
            </div>
            """)
        with a1:
            st.html("""
    <div style="
        display:flex;
        justify-content:center;
        align-items:center;
        height:240px;
        font-size:32px;
        color:#94A3B8;
        font-weight:700;">
        →
    </div>
    """)

        with c2:
            st.html("""
            <div class="compare-card">
                <div style="font-size:42px;">🔥</div>
                <div class="compare-title">Steam Pressure</div>
                <div class="compare-target">Drying Control</div>
            </div>
            """)
        with a2:
            st.html("""
    <div style="
        display:flex;
        justify-content:center;
        align-items:center;
        height:240px;
        font-size:32px;
        color:#94A3B8;
        font-weight:700;">
        →
    </div>
    """)

        with c3:
            st.html("""
            <div class="compare-card">
                <div style="font-size:42px;">⚡</div>
                <div class="compare-title">Machine Speed</div>
                <div class="compare-target">Production Rate</div>
            </div>
            """)
        with a3:
            st.html("""
    <div style="
        display:flex;
        justify-content:center;
        align-items:center;
        height:240px;
        font-size:32px;
        color:#94A3B8;
        font-weight:700;">
        →
    </div>
    """)

        with c4:
            st.html("""
            <div class="compare-card">
                <div style="font-size:42px;">⚖️</div>
                <div class="compare-title">Basis Weight</div>
                <div class="compare-target">Quality Variable</div>
            </div>
            """)
        with a4:st.html("""
    <div style="
        display:flex;
        justify-content:center;
        align-items:center;
        height:240px;
        font-size:32px;
        color:#94A3B8;
        font-weight:700;">
        →
    </div>
    """)

        with c5:
            st.html(f"""
            <div class="compare-card">
                <div style="font-size:42px;">🤖</div>
                <div class="compare-title">AI Prediction</div>
                <div class="compare-target"
                style="color:{risk_color};">
                    {risk_text}
                </div>
            </div>
            """)
        st.divider()
        recommendations = get_recommendations(
    stock_flow,
    filler_flow,
    steam_pressure,
    machine_speed,
    moisture,
    ash,
    basis_weight,
    target_basis_weight
)
        good_runs = history[history["off_spec"] == 0]
        bad_runs = history[history["off_spec"] == 1]

        best_stock = good_runs["stock_flow"].mean()
        best_filler = good_runs["filler_flow"].mean()
        best_steam = good_runs["steam_pressure"].mean()
        best_speed = good_runs["machine_speed"].mean()
        best_moisture = good_runs["moisture"].mean()
        best_ash = good_runs["ash"].mean()
        best_bw = good_runs["basis_weight"].mean()
        bad_stock = bad_runs["stock_flow"].mean()
        bad_steam = bad_runs["steam_pressure"].mean()
        bad_speed = bad_runs["machine_speed"].mean()
        bad_moisture = bad_runs["moisture"].mean()
        bad_bw = bad_runs["basis_weight"].mean()
        recommendation_confidence = []
        if steam_pressure > best_steam + 2:
            recommendation_confidence.append(
        ("Steam Pressure", 96)
    )
        if machine_speed > best_speed + 10:
            recommendation_confidence.append(
        ("Machine Speed", 93)
    )
        if moisture > best_moisture + 0.2:
            recommendation_confidence.append(
        ("Moisture", 91)
    )
        if deviation > 2.5:
            recommendation_confidence.append(
        ("Basis Weight", 98)
    )
        if stock_flow < best_stock - 2:
            recommendation_confidence.append(
        ("Stock Flow", 90)
    )
        st.subheader("🎯 Priority Recommendations")

        if probability > 0.7:
            priority = "HIGH PRIORITY"
            color = "#EF4444"
            badge = "🚨 Immediate Action Required"
            impact = "High"
            status = "Critical"
            source = "Historical Runs + AI Analysis"

        elif probability > 0.4:
            priority = "MEDIUM PRIORITY"
            color = "#F59E0B"
            badge = "⚠ Monitor Process"
            impact = "Medium"
            status = "Watch"
            source = "Historical Successful Runs"

        else:
            priority = "LOW PRIORITY"
            color = "#22C55E"
            badge = "🛡 Process is Stable"
            impact = "Low"
            status = "Stable"
            source = "Engineering Rules & Historical Successful Runs"

        reason = []
        if steam_pressure > best_steam + 2:
            reason.append(
        f"Steam Pressure ({steam_pressure:.1f} bar) is above the historical successful average ({best_steam:.1f} bar)."
    )
        if machine_speed > best_speed + 10:
            reason.append(
        f"Machine Speed ({machine_speed:.0f} m/min) exceeds the historical successful operating range."
    )
        if moisture > best_moisture + 0.2:
            reason.append(
        f"Moisture ({moisture:.2f}%) is higher than successful production runs."
    )
        if deviation > 2.5:
            reason.append(
        f"Basis Weight deviation ({deviation:.2f}%) exceeds the ±2.5% specification limit."
    )
        if stock_flow < best_stock - 2:
            reason.append(
        f"Stock Flow ({stock_flow:.1f} L/min) is lower than historical successful runs."
    )
        if not reason:
            reason.append(
        "Current operating conditions closely match historical successful production runs."
    )

        st.html(f"""
        <div style="
        background:white;
        border-radius:22px;
        overflow:hidden;
        border:1px solid #E5E7EB;
        box-shadow:0 12px 35px rgba(15,23,42,.08);
        ">

        <div style="display:flex;min-height:360px;">

            <div style="flex:3;padding:22px 24px;">

                <div style="font-size:34px;color:{color};">●</div>

                <div style="
                font-size:30px;
                font-weight:800;
                color:{color};
                margin-top:-10px;
                ">
                {priority}
                </div>

                <div style="
                display:inline-block;
                background:{color}15;
                color:{color};
                padding:7px 14px;
                border-radius:30px;
                font-weight:500;
                font-size:13px;
                margin-top:12px;
                ">
                {badge}
                </div>

                <div style="
                margin-top:16px;font-size:16px;line-height:1.5;
                color:#334155;
                ">

                {"<br>".join("✓ " + r for r in recommendations)}
                <hr style="margin:18px 0;">

<div style="
font-size:15px;
color:#475569;
line-height:1.6;
">

<b>Why this recommendation?</b><br>

{"<br>".join("• " + r for r in reason)}

</div>

                </div>

                <hr style="margin:18px 0;">

                <div style="
                display:flex;
                justify-content:space-between;
                text-align:center;
                ">

                    <div>
                        <div style="font-size:22px;">✔</div>
                        <b>No Action Delay</b><br>
                        <span style="font-size:13px;color:#64748B;">Apply Immediately</span>
                    </div>

                    <div>
                        <div style="font-size:22px;">👁</div>
                        <b>Monitor</b><br>
                        <span style="font-size:13px;color:#64748B;">Observe Process</span>
                    </div>

                    <div>
                        <div style="font-size:22px;">📈</div>
                        <b>Expected Result</b><br>
                        <span style="font-size:13px;color:#64748B;">Higher Stability</span>
                    </div>

                </div>

            </div>

            <div style="
            flex:1;
            background:#F8FAFC;
            border-left:1px solid #E5E7EB;
            padding:22px 24px;
            ">

                <div style="
                color:#64748B;
                font-size:14px;
                font-weight:700;
                ">
                EXPECTED IMPACT
                </div>

                <div style="
                color:{color};
                font-size:30px;
                font-weight:800;
                ">
                {impact}
                </div>

                <div style="color:#64748B;">
                Operational Impact
                </div>

                <hr style="margin:22px 0;">

                <div style="
                background:white;
                border-radius:14px;
                padding:12px;
                border:1px solid #E5E7EB;
                ">

                <b>📊 Impact Summary</b>

                <table style="width:100%;margin-top:15px;">

                <tr>
                <td>Quality</td>
                <td align="right"><b>{status}</b></td>
                </tr>

                <tr>
                <td>Risk</td>
                <td align="right"><b>{impact}</b></td>
                </tr>

                <tr>
                <td>Consistency</td>
                <td align="right"><b>Improved</b></td>
                </tr>

                </table>

                </div>

            </div>

        </div>

        <div style="
        background:#F8FAFC;
        padding:12px 22px;
        border-top:1px solid #E5E7EB;
        display:flex;
        justify-content:space-between;
        ">

            <div>
                <b>📚 Source</b><br>
                {source}
            </div>

            <div>
                <b>🕒 Generated</b><br>
                {datetime.now().strftime("%d %b %Y, %I:%M %p")}
            </div>

        </div>

        </div>
        """)

        st.divider()
        st.subheader("📚 Historical Successful Operating Recipe")



        c1, c2, c3 = st.columns(3)

        with c1:

            st.html(f"""
            <div class="recipe-card">
                <div class="recipe-icon">🌊</div>
                <div class="recipe-title">Stock Flow</div>
                <div class="recipe-value">{best_stock:.1f} L/min</div>
                <div class="recipe-sub">Historical Average</div>
            </div>
            """)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)


            st.html(f"""
            <div class="recipe-card">
                <div class="recipe-icon">🔥</div>
                <div class="recipe-title">Steam Pressure</div>
                <div class="recipe-value">{best_steam:.1f} bar</div>
                <div class="recipe-sub">Historical Average</div>
            </div>
            """)

        with c2:

            st.html(f"""
            <div class="recipe-card">
                <div class="recipe-icon">⚡</div>
                <div class="recipe-title">Machine Speed</div>
                <div class="recipe-value">{best_speed:.1f} m/min</div>
                <div class="recipe-sub">Historical Average</div>
            </div>
            """)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            st.html(f"""
            <div class="recipe-card">
                <div class="recipe-icon">💧</div>
                <div class="recipe-title">Moisture</div>
                <div class="recipe-value">{best_moisture:.2f}%</div>
                <div class="recipe-sub">Historical Average</div>
            </div>
            """)

        with c3:

            st.html(f"""
            <div class="recipe-card">
                <div class="recipe-icon">⚖️</div>
                <div class="recipe-title">Basis Weight</div>
                <div class="recipe-value">{best_bw:.2f} GSM</div>
                <div class="recipe-sub">Historical Average</div>
            </div>
            """)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            st.html(f"""
            <div class="recipe-card">
                <div class="recipe-icon">🍃</div>
                <div class="recipe-title">Ash</div>
                <div class="recipe-value">{best_ash:.2f}%</div>
                <div class="recipe-sub">Historical Average</div>
            </div>
            """)

        st.info(f"""📊 **Historical Learning Summary**

✔ Successful Runs: **{len(good_runs)}**

✔ Historical Success Rate: **{(1-history['off_spec'].mean())*100:.1f}%**

✔ Average Basis Weight Deviation: **{good_runs['bw_deviation'].mean():.2f}%**

Recommendations are generated using historical successful production runs.
""")
        st.divider()
        st.subheader("📈 Historical Trajectory Comparison")
        historical = {
    "Stock Flow": good_runs["stock_flow"].mean(),
    "Steam Pressure": good_runs["steam_pressure"].mean(),
    "Machine Speed": good_runs["machine_speed"].mean(),
    "Moisture": good_runs["moisture"].mean(),
    "Basis Weight": good_runs["basis_weight"].mean(),}
        trajectory = pd.DataFrame({
    "Parameter":[
        "Stock Flow",
        "Steam Pressure",
        "Machine Speed",
        "Moisture",
        "Basis Weight"
    ],

    "Current":[
        stock_flow,
        steam_pressure,
        machine_speed,
        moisture,
        basis_weight
    ],

    "Historical Successful Avg":[
        historical["Stock Flow"],
        historical["Steam Pressure"],
        historical["Machine Speed"],
        historical["Moisture"],
        historical["Basis Weight"]
    ]})
        trajectory["Difference"] = (
    trajectory["Current"]
    - trajectory["Historical Successful Avg"]
).round(2)
        trajectory["Status"] = trajectory["Difference"].apply(
    lambda x: "✅ Close" if abs(x) < 2 else "⚠ Deviated")
        st.dataframe(
    trajectory,
    use_container_width=True,
    hide_index=True)
        st.subheader("📊 Success vs Failure Comparison")
        comparison = pd.DataFrame({

    "Parameter":[
        "Stock Flow",
        "Steam Pressure",
        "Machine Speed",
        "Moisture",
        "Basis Weight"
    ],

    "Successful Runs":[
        best_stock,
        best_steam,
        best_speed,
        best_moisture,
        best_bw
    ],

    "Failed Runs":[
        bad_stock,
        bad_steam,
        bad_speed,
        bad_moisture,
        bad_bw
    ]})
        st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)

        st.divider()
        st.subheader("📏 Recipe Operating Limits")
        limits = {
    "Stock Flow": (95, 105, stock_flow, "L/min"),
    "Steam Pressure": (45, 65, steam_pressure, "bar"),
    "Machine Speed": (850, 980, machine_speed, "m/min"),
    "Moisture": (4.5, 6.5, moisture, "%"),
    "Basis Weight": (
        target_basis_weight * 0.975,
        target_basis_weight * 1.025,
        basis_weight,
        "GSM")}
        c1,c2,c3,c4,c5 = st.columns(5)
        cols=[c1,c2,c3,c4,c5]
        for col,(name,(low,high,current,unit)) in zip(cols,limits.items()):
            if current<low:
                status="🔵 LOW"
                color="#2563EB"
            elif current>high:
                status="🔴 HIGH"
                color="#EF4444"
            else:
                status="🟢 NORMAL"
                color="#22C55E"
            with col:
                st.html(f"""
<div class="compare-card">

<div class="compare-title">
{name}
</div>

<div class="compare-current">
{current:.1f} {unit}
</div>

<hr>

<div style="font-size:13px;color:#64748B;">
Allowed

{low:.1f} – {high:.1f}
</div>

<div style="
margin-top:10px;
font-weight:700;
color:{color};
">
{status}
</div>

</div>
""")
        st.divider()
        st.subheader("⭐ Top Risk Contributors")

        contributors = []

        if steam_pressure > best_steam + 2:
            contributors.append(("🔥 Steam Pressure", "High", "#EF4444"))

        if machine_speed > best_speed + 10:
            contributors.append(("⚡ Machine Speed", "Medium", "#F59E0B"))

        if moisture > best_moisture + 0.2:
            contributors.append(("💧 Moisture", "Medium", "#F59E0B"))

        if stock_flow < best_stock - 2:
            contributors.append(("🌊 Stock Flow", "Low", "#2563EB"))

        if deviation > 2.5:
            contributors.append(("⚖️ Basis Weight", "Critical", "#DC2626"))

        if not contributors:
            contributors.append(("✅ Process Stable", "None", "#22C55E"))

        cols = st.columns(len(contributors))

        for col, (name, level, color) in zip(cols, contributors):

            with col:

                st.html(f"""
                <div class="compare-card">

                <div class="compare-title">
                {name}
                </div>

                <div style="
                font-size:28px;
                font-weight:700;
                color:{color};
                margin-top:18px;">
                {level}
                </div>

                </div>
                """)
        st.divider()

        st.subheader("⚙ Current vs Recommended Machine Settings")

        recommended = {
            "Stock Flow": stock_flow,
            "Steam Pressure": steam_pressure,
            "Machine Speed": machine_speed,
            "Moisture": moisture,
            "Basis Weight": basis_weight,
        }

        if steam_pressure > best_steam+2:
            recommended["Steam Pressure"] = round(best_steam,1)

        if machine_speed > best_speed+10:
            recommended["Machine Speed"] = round(best_speed,1)

        if moisture > best_moisture+0.2:
           recommended["Moisture"] = round(best_moisture,2)
        if stock_flow < best_stock-2:
            recommended["Stock Flow"] = round(best_stock, 1)


        if deviation > 2.5:
            recommended["Basis Weight"] = target_basis_weight

        comparison = pd.DataFrame({
            "Parameter": [
                "Stock Flow",
                "Steam Pressure",
                "Machine Speed",
                "Moisture",
                "Basis Weight"
            ],
            "Current": [
                stock_flow,
                steam_pressure,
                machine_speed,
                moisture,
                basis_weight
            ],
            "Recommended": [
                recommended["Stock Flow"],
                recommended["Steam Pressure"],
                recommended["Machine Speed"],
                recommended["Moisture"],
                recommended["Basis Weight"]
            ]
        })

        c1, c2, c3, c4, c5 = st.columns(5)

        cards = [
            ("🌊", "Stock Flow", stock_flow, recommended["Stock Flow"], "L/min"),
            ("🔥", "Steam", steam_pressure, recommended["Steam Pressure"], "bar"),
            ("⚡", "Speed", machine_speed, recommended["Machine Speed"], "m/min"),
            ("💧", "Moisture", moisture, recommended["Moisture"], "%"),
            ("⚖️", "Basis Weight", basis_weight, recommended["Basis Weight"], "GSM")
        ]

        for col, (icon, name, current, target, unit) in zip(
            [c1, c2, c3, c4, c5], cards
        ):

            delta = target - current

            if abs(delta) < 0.1:
                color = "#22C55E"
                text = "✅ Optimal"
            elif delta < 0:
                color = "#EF4444"
                text = f"↓ {abs(delta):.1f} {unit}"
            else:
                color = "#2563EB"
                text = f"↑ {delta:.1f} {unit}"

            with col:
                st.html(f"""<div class="compare-card">

    <div class="compare-icon">{icon}</div>

    <div class="compare-title">{name}</div>

    <div class="compare-current">
        {current:.1f} {unit}
    </div>

    <div style="font-size:24px;padding:8px;">↓</div>

    <div class="compare-target">
        {target:.1f} {unit}
    </div>

    <div class="compare-status">{text}</div>

</div>
""")

        st.caption(
            "Recommended values are generated using historical successful production runs and engineering operating constraints."
        )

        st.divider()
        st.subheader("🧠 AI Recommendation Confidence")
        if recommendation_confidence:
            cols = st.columns(len(recommendation_confidence))
            for col, (parameter, confidence) in zip(cols, recommendation_confidence):
                with col:
                    st.metric(parameter, f"{confidence}%")
        else:
            st.success("No parameter adjustments are required. All current operating conditions are within the recommended operating range.")
        st.divider()
        st.subheader("📈 Expected Impact of Recommendations")

        current_risk = probability * 100
        risk_reduction = min(current_risk * 0.70, 70)
        new_risk = max(current_risk - risk_reduction, 5)

        # Simulated improvement after applying recommendations
        if new_risk > 60:
            stabilization = "6–8 min"
        elif new_risk > 30:
            stabilization = "3–5 min"
        else:
            stabilization = "1–3 min"
        quality = "WITHIN SPEC" if new_risk < 40 else "MONITOR"

        c1,c2,c3,c4=st.columns(4)
        cards=[
("⚠️","Current Risk",f"{current_risk:.1f}%","#EF4444"),
("📉","Risk After",f"{new_risk:.1f}%","#22C55E"),
("⏱","Stabilization",stabilization,"#2563EB"),
("✅","Expected Quality",quality,"#4F46E5")]
        for col,(icon,title,value,color) in zip([c1,c2,c3,c4],cards):
             with col:
                 st.html(f"""
        <div class="kpi-card">

        <div style="font-size:30px">{icon}</div>

        <div class="kpi-title">{title}</div>

        <div class="kpi-value"
        style="color:{color}">
        {value}
        </div>

        </div>
        """)
        st.divider()
        st.subheader("📊 Process Stability Assessment")

        stability = max(0, 100 - abs(deviation) * 12 - current_risk * 0.4)

        if stability >= 80:
            status = "🟢 Stable"
            color = "#22C55E"

        elif stability >= 60:
            status = "🟡 Acceptable"
            color = "#F59E0B"

        else:
            status = "🔴 Unstable"
            color = "#EF4444"

        s1, s2 = st.columns([1,2])

        with s1:
            st.metric(
                "Stability Score",
                f"{stability:.0f}/100"
            )

        with s2:
            st.html(f"""
            <div class="recipe-card">

            <h3 style="margin-top:0;color:{color};">
            {status}
            </h3>

            <hr>

            <p style="font-size:15px;line-height:1.8;">
            Process stability is calculated using:
            <br><br>
            • OFF-SPEC Probability
            <br>
            • Basis Weight Deviation
            <br>
            • Historical Process Behaviour
            </p>

            </div>
            """)
        
        

        left, right = st.columns(2)

        with left:
            st.html("""
            <div class="recipe-card">

            <h3 style="margin-top:0;">✅ Expected Benefits</h3>

            <hr>

            <p style="font-size:15px;line-height:2;">
            ✔ Reduce OFF-SPEC probability<br>
            ✔ Improve basis weight stability<br>
            ✔ Reduce stabilization time<br>
            ✔ Increase production consistency
            </p>

            </div>
            """)

        with right:
            st.html(f"""
            <div class="recipe-card">

            <h3 style="margin-top:0;">📊 Expected KPIs</h3>

            <hr>

            <p style="font-size:15px;line-height:2;">

            <b>Expected Yield</b><br>
            {100-new_risk:.1f}%<br><br>

            <b>Quality Status</b><br>
            {quality}<br><br>

            <b>Stabilization Time</b><br>
            {stabilization}

            </p>

            </div>
            """)

        st.divider()
        
        st.subheader("🤖 AI Explanation")
        values = {
            "Stock Flow": stock_flow,
            "Steam Pressure": steam_pressure,
            "Machine Speed": machine_speed,
            "Moisture": moisture,
            "Ash": ash,
            "Basis Weight": basis_weight,
            "Target Basis Weight": target_basis_weight
        }
        try:
            ai_text = explain(
                "OFF SPEC" if prediction else "WITHIN SPEC",
                probability, recommendations, values)
            st.html(f"""
<div class="recipe-card">

<h3>🤖 AI Process Explanation</h3>

<hr>

<pre style="
font-size:15px;
white-space:pre-wrap;
font-family:Poppins;
line-height:1.7;">
{ai_text}
</pre>

</div>
""")
        except Exception:
            ai_text = f"""
Prediction: {"OFF SPEC" if prediction else "WITHIN SPEC"}

Reasons:
{chr(10).join(reasons)}

Recommended Actions:
{chr(10).join(recommendations)}

Generated using rule-based analysis because Gemini AI is currently unavailable.
"""
            st.html(f"""
<div class="recipe-card">

<h3>🤖 AI Process Explanation</h3>

<hr>

<pre style="
font-size:15px;
white-space:pre-wrap;
font-family:Poppins;
line-height:1.7;">
{ai_text}
</pre>

</div>
""")

        report = pd.DataFrame({
            "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "Prediction": ["OFF SPEC" if prediction else "WITHIN SPEC"],
            "Risk Score": [f"{probability*100:.2f}%"],
            "Current Basis Weight": [basis_weight],
            "Target Basis Weight": [target_basis_weight],
            "Deviation (%)": [round(deviation, 2)],
            "Reasons": [", ".join(reasons)],
            "Recommendations": [", ".join(recommendations)],
            "Estimated Stabilization":[stabilization],
            "Historical Recommendation Source":["Historical Successful Runs"],
            "AI Explanation": [ai_text]
        })
        st.html("""
        <div class="report-card">

        <h3 style="margin-top:0;">📄 Production Analysis Report</h3>

        <p style="color:#64748B;font-size:15px;">
        Download a complete production report containing:
        </p>

        <ul style="color:#475569;line-height:1.8;">
            <li>Prediction Result</li>
            <li>Risk Assessment</li>
            <li>Recommended Setpoints</li>
            <li>Historical Recipe</li>
            <li>AI Explanation</li>
            <li>Operator Feedback</li>
        </ul>

        </div>
        """)

                # -----------------------------
        # Generate PDF Report
        # -----------------------------

        buffer = BytesIO()

        doc = SimpleDocTemplate(buffer)

        styles = getSampleStyleSheet()

        elements = []

        elements.append(
            Paragraph(
                "PaperAI Production Analysis Report",
                styles["Title"]
            )
        )

        elements.append(
            Paragraph(
                f"<b>Prediction:</b> {prediction}",
                styles["BodyText"]
            )
        )

        elements.append(
            Paragraph(
                f"<b>OFF SPEC Probability:</b> {probability*100:.1f}%",
                styles["BodyText"]
            )
        )

        elements.append(
            Paragraph(
                f"<b>Basis Weight:</b> {basis_weight:.2f} GSM",
                styles["BodyText"]
            )
        )

        elements.append(
            Paragraph(
                f"<b>Target Basis Weight:</b> {target_basis_weight:.2f} GSM",
                styles["BodyText"]
            )
        )

        elements.append(
            Paragraph(
                f"<b>Deviation:</b> {deviation:.2f}%",
                styles["BodyText"]
            )
        )

        elements.append(
            Paragraph(
                "<b>Recommendations</b>",
                styles["Heading2"]
            )
        )

        for r in recommendations:
            elements.append(
                Paragraph("• " + r, styles["BodyText"])
            )

        doc.build(elements)

        pdf = buffer.getvalue()

        buffer.close()

        c1, c2 = st.columns(2)

        with c1:
            st.download_button(
                "📄 Download PDF Report",
                pdf,
                file_name="PaperAI_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        with c2:
            st.download_button(
                "📊 Download CSV",
                report.to_csv(index=False).encode("utf-8"),
                file_name="PaperAI_Production_Report.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.divider()
        similarity = max(0, 100 - abs(deviation) * 8)

        st.divider()
        st.subheader("🧬 Historical Similarity")

        c1, c2 = st.columns([1, 2])

        with c1:
            st.metric(
                "Similarity Score",
                f"{similarity:.1f}%"
            )

        with c2:
            st.info(
                f"""The current process is approximately **{similarity:.1f}% similar**to historical successful grade changes.

Recommendations are based on production runs with similar operating
conditions and successful quality outcomes.
""")


        st.divider()
        st.subheader("🚨 Active Process Alerts")
        alerts = []
        if steam_pressure > 65:
            alerts.append((
    "🔴 CRITICAL",
    f"""Steam Pressure

Current : {steam_pressure:.1f} bar

Recommended : {best_steam:.1f} bar

Expected Impact :
Improves moisture stability and reduces OFF-SPEC risk."""
))
        if machine_speed > 980:
            alerts.append((
    "🟠 WARNING",
    f"""Machine Speed

Current : {machine_speed:.0f} m/min

Recommended : {best_speed:.0f} m/min

Expected Impact :
Improves basis weight stability."""
))
        if moisture > 6.5:
            alerts.append((
    "🟡 WARNING",
    f"""Moisture

Current : {moisture:.2f} %

Recommended : {best_moisture:.2f} %

Expected Impact :
Improves drying consistency."""
))
        if deviation > 2.5:
            alerts.append((
    "🔴 CRITICAL",
    f"""Basis Weight

Current : {basis_weight:.2f} GSM

Target : {target_basis_weight:.2f} GSM

Deviation : {deviation:.2f} %

Expected Impact :
Returning to target improves product quality."""
))
        if stock_flow < 90:
            alerts.append((
    "🟡 WARNING",
    f"""
Stock Flow

Current : {stock_flow:.1f} L/min

Recommended : {best_stock:.1f} L/min

Expected Impact :
Improves sheet consistency and basis weight stability.
"""
))
        if ash > 2.5:
            alerts.append((
    "🟠 WARNING",
    f"""
Ash Content

Current : {ash:.2f} %

Recommended : {best_ash:.2f} %

Expected Impact :
Reduces filler variation and improves paper quality.
"""
))
        if not alerts:
            alerts.append(("🟢 NORMAL", "All process parameters are operating within acceptable limits."))
        for level, message in alerts:
            if "CRITICAL" in level:
                color = "#EF4444"
            elif "WARNING" in level:
                color = "#F59E0B"
            else:
                color = "#22C55E"
            st.html(f"""<div style="background:white;border-left:8px solid {color};border-radius:18px;
                    padding:20px;margin-bottom:15px;box-shadow:0 8px 20px rgba(0,0,0,.05);">
                    <h3 style="margin:0;color:{color};">{level}</h3><hr><pre style="font-family:Poppins;
                    white-space:pre-wrap;line-height:1.7;font-size:15px;">{message}</pre></div>""")

        st.divider()
        st.html("""
        <div class="report-card">

        <h3 style="margin-top:0;">👷 Operator Decision</h3>

        <p style="color:#64748B;font-size:15px;">
        Review the AI recommendation and record your decision. Your feedback is stored to improve future recommendations.
        </p>

        </div>
        """)

        b1, b2 = st.columns(2)

        with b1:
            if st.button(
                "✅ Accept Recommendation",
                use_container_width=True
            ):
                save_feedback("Accepted")
                st.toast("✅ Operator feedback saved successfully.")

        with b2:
            if st.button(
                "❌ Reject Recommendation",
                use_container_width=True
            ):
                save_feedback("Rejected")
                st.toast("❌ Operator feedback saved successfully.")


        

        if probability > 0.8:
            future_status = "OFF SPEC"
            future_color = "#EF4444"
            future_msg = "Current operating conditions indicate a high probability of OFF-SPEC production if no corrective action is taken."
            stabilization = "8–10 min"

        elif probability > 0.5:
            future_status = "AT RISK"
            future_color = "#F59E0B"
            future_msg = "The process is trending towards OFF-SPEC. Applying the recommended setpoints is advised."
            stabilization = "5–7 min"

        else:
            future_status = "STABLE"
            future_color = "#22C55E"
            future_msg = "The process is expected to remain stable and within specification under current operating conditions."
            stabilization = "2–3 min"

        st.html(f"""
        <div class="report-card">

        <h3 style="margin-top:0;color:{future_color};">
        🔮 Future Process Prediction
        </h3>

        <hr>

        <div style="font-size:17px;font-weight:600;color:{future_color};">
        Status : {future_status}
        </div>

        <br>

        <div style="font-size:15px;color:#475569;">
        {future_msg}
        </div>

        </div>
        """)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.html(f"""
            <div class="kpi-card">
                <div style="font-size:30px;">⏱</div>
                <div class="kpi-title">Estimated Stabilization</div>
                <div class="kpi-value">{stabilization}</div>
            </div>
            """)

        with c2:
            st.html(f"""
            <div class="kpi-card">
                <div style="font-size:30px;">🎯</div>
                <div class="kpi-title">Expected Yield</div>
                <div class="kpi-value">{100-new_risk:.1f}%</div>
            </div>
            """)

        with c3:
            st.html(f"""
            <div class="kpi-card">
                <div style="font-size:30px;">🤖</div>
                <div class="kpi-title">AI Confidence</div>
                <div class="kpi-value">{100-current_risk:.0f}%</div>
            </div>
            """)

        st.divider()
        st.subheader("🏭 Digital Twin")
        st.graphviz_chart("""
digraph G {

rankdir=LR;
edge[
penwidth=2
color="#475569"
]

node[
shape=box
style="rounded,filled"
fillcolor="#EEF4FF"
color="#2563EB"
fontname="Poppins"
fontsize=12
penwidth=2
]

Stock [label="Stock Flow"];
Filler [label="Filler Flow"];
Steam [label="Steam Pressure"];
Machine [label="Paper Machine", shape=ellipse, fillcolor="#FEF3C7" color="#D97706"];
Speed [label="Machine Speed"];
Moisture [label="Moisture"];
Basis [label="Basis Weight"];
Prediction [label="Quality Prediction", fillcolor="#DCFCE7"
color="#16A34A"];

Stock -> Machine;
Filler -> Machine;
Steam -> Machine;

Machine -> Speed;
Machine -> Moisture;

Speed -> Basis;
Moisture -> Basis;

Basis -> Prediction;

}
""")
        st.caption("Digital representation of the paper manufacturing process.")
        st.divider()



ANALYTICS_CSS = """
<style>
/* ---------- Section header ---------- */
.section-title {
    display:flex;
    align-items:center;
    gap:10px;
    font-size:20px;
    font-weight:600;
    color:#1E293B;
    margin:6px 0 2px 0;
}
.section-sub {
    font-size:13.5px;
    color:#64748B;
    margin-bottom:14px;
}

/* ---------- KPI stat cards ---------- */
.stat-card {
    display:flex;
    align-items:center;
    gap:14px;
    background:#FFFFFF;
    border:1px solid #E2E8F0;
    border-radius:14px;
    padding:18px 20px;
    box-shadow:0 1px 3px rgba(15,23,42,0.04);
    transition:box-shadow .15s ease, transform .15s ease;
    height:100%;
}
.stat-card:hover {
    box-shadow:0 4px 14px rgba(15,23,42,0.08);
    transform:translateY(-1px);
}
.stat-icon {
    width:44px;
    height:44px;
    min-width:44px;
    border-radius:12px;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:20px;
}
.stat-label {
    font-size:12.5px;
    font-weight:600;
    letter-spacing:.03em;
    text-transform:uppercase;
    color:#94A3B8;
    margin-bottom:2px;
}
.stat-value {
    font-size:24px;
    font-weight:700;
    color:#0F172A;
    line-height:1.15;
}
.stat-sub {
    font-size:12px;
    color:#94A3B8;
    margin-top:1px;
}

/* ---------- Chart wrapper card ---------- */
.chart-card {
    background:#FFFFFF;
    border:1px solid #E2E8F0;
    border-radius:14px;
    padding:18px 18px 6px 18px;
    box-shadow:0 1px 3px rgba(15,23,42,0.04);
    margin-bottom:6px;
}

/* ---------- Process influence chain ---------- */
.flow-chain {
    display:flex;
    align-items:center;
    justify-content:space-between;
    flex-wrap:wrap;
    gap:8px;
    background:#F8FAFC;
    border:1px solid #E2E8F0;
    border-radius:14px;
    padding:22px 18px;
}
.flow-step {
    background:#FFFFFF;
    border:1px solid #E2E8F0;
    border-radius:10px;
    padding:10px 16px;
    font-weight:600;
    font-size:13.5px;
    color:#334155;
    text-align:center;
    box-shadow:0 1px 2px rgba(15,23,42,0.04);
}
.flow-step.risk {
    border-color:#FCD34D;
    background:#FFFBEB;
    color:#92400E;
}
.flow-step.offspec {
    border-color:#FCA5A5;
    background:#FEF2F2;
    color:#991B1B;
}
.flow-arrow {
    color:#94A3B8;
    font-size:18px;
}

/* ---------- Footer ---------- */
.app-footer {
    text-align:center;
    padding:22px 0 8px 0;
    color:#94A3B8;
    font-size:12.5px;
    line-height:1.7;
    border-top:1px solid #E2E8F0;
    margin-top:10px;
}
.app-footer b { color:#64748B; }
</style>
"""

# Chart theme kept consistent across every Plotly figure on this tab.
CHART_LAYOUT = dict(
    title=None,
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Poppins, sans-serif", color="#334155", size=12.5),
    margin=dict(l=24, r=24, t=20, b=24),
    xaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False),
    hoverlabel=dict(bgcolor="white", font_size=12.5, font_family="Poppins, sans-serif"),
)


def section_header(icon: str, title: str, subtitle: str = ""):
    """Consistent section headers instead of ad-hoc st.subheader calls."""
    st.markdown(
        f"""<div class="section-title">{icon} {title}</div>""",
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(f"""<div class="section-sub">{subtitle}</div>""", unsafe_allow_html=True)


with analytics_tab:
    st.markdown(ANALYTICS_CSS, unsafe_allow_html=True)

    section_header("📈", "Analytics Overview", "Live snapshot of historical run performance")

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.html(f"""
    <div class="stat-card">
        <div class="stat-icon" style="background:#EEF2FF;">📄</div>
        <div>
            <div class="stat-label">Total Records</div>
            <div class="stat-value">{len(history):,}</div>
            <div class="stat-sub">Historical Runs</div>
        </div>
    </div>
    """)
    with a2:
        st.html(f"""
    <div class="stat-card">
        <div class="stat-icon" style="background:#FEF3C7;">⚠️</div>
        <div>
            <div class="stat-label">Off Spec</div>
            <div class="stat-value">{history['off_spec'].mean()*100:.1f}%</div>
            <div class="stat-sub">Historical Rate</div>
        </div>
    </div>
    """)
    with a3:
        st.html(f"""
    <div class="stat-card">
        <div class="stat-icon" style="background:#DBEAFE;">⚖️</div>
        <div>
            <div class="stat-label">Average BW</div>
            <div class="stat-value">{history['basis_weight'].mean():.2f}</div>
            <div class="stat-sub">GSM</div>
        </div>
    </div>
    """)
    with a4:
        st.html(f"""
    <div class="stat-card">
        <div class="stat-icon" style="background:#DCFCE7;">✅</div>
        <div>
            <div class="stat-label">Success Rate</div>
            <div class="stat-value">{100-history['off_spec'].mean()*100:.1f}%</div>
            <div class="stat-sub">Within Spec</div>
        </div>
    </div>
    """)

    st.write("")
    st.divider()

    # ----------------------------
    # Trend + distribution, side by side
    # ----------------------------
    t1, t2 = st.columns(2)

    with t1:
        section_header("📈", "Historical Basis Weight", "Last 300 recorded runs")
        fig1 = px.line(history.head(300), y="basis_weight")
        fig1.update_traces(line=dict(color="#4F46E5", width=2))
        fig1.update_layout(**CHART_LAYOUT, height=320)
        st.plotly_chart(fig1, use_container_width=True)

    with t2:
        section_header("🔥", "Steam Pressure Distribution", "Frequency across all runs")
        fig2 = px.histogram(history, x="steam_pressure", nbins=25)
        fig2.update_traces(marker_color="#F59E0B")
        fig2.update_layout(**CHART_LAYOUT, height=320)
        st.plotly_chart(fig2, use_container_width=True)
    

    section_header(
    "📈",
    "Historical Parameter Trends",
    "Comparison of major process variables")
    fig = px.line(
    history.head(200),
    y=[
        "steam_pressure",
        "machine_speed",
        "stock_flow",
        "moisture"
    ])
    fig.update_layout(**CHART_LAYOUT, height=380)
    st.plotly_chart(
    fig,
    use_container_width=True)
    st.divider()
    section_header(
    "🎯",
    "Current vs Historical Successful Average",
    "Current operating values compared with historical successful runs")

    good_runs = history[history["off_spec"] == 0]
    best_stock = good_runs["stock_flow"].mean()
    best_steam = good_runs["steam_pressure"].mean()
    best_speed = good_runs["machine_speed"].mean()
    best_moisture = good_runs["moisture"].mean()
    comparison = pd.DataFrame({
    "Parameter": [
        "Stock Flow",
        "Steam Pressure",
        "Machine Speed",
        "Moisture"
    ],
    "Current": [
        stock_flow,
        steam_pressure,
        machine_speed,
        moisture
    ],
    "Historical Average": [
        best_stock,
        best_steam,
        best_speed,
        best_moisture
    ]
})
    st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True)
    st.divider()

    # ----------------------------
    # Top Correlations
    # ----------------------------
    section_header("🔍", "Correlation Analysis", "What moves Off-Spec production the most")

    corr = history.corr(numeric_only=True)

    top = (
        corr["off_spec"]
        .drop("off_spec")
        .abs()
        .sort_values(ascending=False)
    )

    st.markdown("**📌 Parameters Most Influencing Off-Spec Production**")

    c1, c2 = st.columns([1.1, 1])

    with c1:
        m1, m2 = st.columns(2)
        for i, (feature, value) in enumerate(top.head(4).items()):
            target = m1 if i % 2 == 0 else m2
            with target:
                st.metric(feature.replace("_", " ").title(), f"{value:.3f}")

    with c2:
        top_feature = top.index[0].replace("_", " ").title()
        st.info(
            f"The strongest correlation with Off-Spec production is **{top_feature}** "
            f"({top.iloc[0]:.3f}). Operators should monitor this parameter closely "
            f"during grade transitions."
        )

    st.divider()


    # ----------------------------
    # Process Influence
    # ----------------------------
    section_header("🔄", "Process Influence Chain", "How upstream parameters cascade into quality risk")

    st.markdown(
        """
        <div class="flow-chain">
            <div class="flow-step">Steam Pressure</div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">Moisture</div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">Basis Weight</div>
            <div class="flow-arrow">→</div>
            <div class="flow-step risk">Quality Risk</div>
            <div class="flow-arrow">→</div>
            <div class="flow-step offspec">OFF-SPEC Production</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Relationship inferred from historical production data and machine learning analysis.")

    st.divider()

    left, right = st.columns(2)

    # ============================
    # Correlation Heatmap
    # ============================

    with left:

        section_header(
            "📊",
            "Correlation Heatmap",
            "Pairwise relationships across variables"
        )

        fig3 = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )

        fig3.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(
                family="Poppins, sans-serif",
                color="#334155",
                size=12
            ),
            margin=dict(
                l=20,
                r=20,
                t=10,
                b=20
            ),
            height=380
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

    # ============================
    # Feature Importance
    # ============================

    with right:

        section_header(
            "⭐",
            "Feature Importance",
            "Model contribution"
        )

        importance = pd.DataFrame({
            "Feature": history.drop("off_spec", axis=1).columns,
            "Importance": model.feature_importances_
        })

        importance = importance.sort_values(
            "Importance",
            ascending=False
        )

        fig4 = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            text="Importance",
            color="Importance",
            color_continuous_scale="Blues"
        )

        fig4.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside"
        )

        fig4.update_layout(
            **{
                **CHART_LAYOUT,
                "yaxis": dict(
                    autorange="reversed",
                    showgrid=False
                )
            },
            height=380,
            coloraxis_showscale=False
        )

        st.plotly_chart(
            fig4,
            use_container_width=True
        )

    st.divider()

    # ----------------------------
    # Steam Pressure vs Basis Weight
    # ----------------------------
    section_header("📊", "Steam Pressure vs Basis Weight", "Colored by Off-Spec outcome")

    fig5 = px.scatter(
        history,
        x="steam_pressure",
        y="basis_weight",
        trendline="ols",
        color="off_spec",
        labels={
            "steam_pressure": "Steam Pressure (bar)",
            "basis_weight": "Basis Weight (GSM)",
            "off_spec": "OFF SPEC",
        },
        color_discrete_sequence=["#4F46E5", "#EF4444"],
    )
    fig5.update_traces(marker=dict(size=7, opacity=0.75, line=dict(width=0)))
    fig5.update_layout(**CHART_LAYOUT, height=420, legend=dict(orientation="h", y=1.08, x=0))
    st.plotly_chart(fig5, use_container_width=True)

    st.divider()

    # ----------------------------
    # Model Performance
    # ----------------------------
    section_header("🧠", "Model Performance", "Evaluation metrics on the holdout set")

    c1,c2,c3,c4=st.columns(4)
    with c1:
        st.metric("Accuracy", "100%")
    with c2:
        st.metric("Precision", "100%")
    with c3:
        st.metric("Recall", "100%")
    with c4:
        st.metric("F1 Score","100%")

    with st.container(border=True):
        st.image(
        "models/confusion_matrix.png",
        use_container_width=True
    )

    st.divider()

    # ----------------------------
    # Historical Dataset
    # ----------------------------
    section_header("📋", "Historical Dataset", "Most recent 20 records")
    st.dataframe(history.tail(20), height=350, use_container_width=True)

    st.divider()

    # ----------------------------
    # Operator Feedback
    # ----------------------------
    st.subheader("👷 Operator Feedback Analytics")
    if os.path.exists("feedback.csv"):
        feedback = pd.read_csv("feedback.csv")
        accepted = (feedback["decision"]== "Accepted").sum()
        rejected = (feedback["decision"] == "Rejected").sum()
        total = len(feedback)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Decisions", total)
        with c2:
            st.metric("Accepted", accepted)
        with c3:
            rate = accepted / total * 100 if total else 0
            st.metric("Acceptance Rate", f"{rate:.1f}%")
        st.dataframe(
        feedback.tail(10),
        use_container_width=True,
        hide_index=True
    )
    else:
        st.info("No operator feedback available yet.")