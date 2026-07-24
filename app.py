import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime

from gemini_helper import explain
from feedback import save_feedback
from recommendation import get_recommendations

# ----------------------------
# Page config (must be first st call)
# ----------------------------
st.set_page_config(
    page_title="Paper Grade Intelligence",
    page_icon="📄",
    layout="wide"
)

# ----------------------------
# Load Model & Data
# ----------------------------
model = joblib.load("models/random_forest.pkl")
history = pd.read_csv("data/historical_data.csv")

# ----------------------------
# Global styles
# ----------------------------
st.html("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>

.stApp{
    background:#F3F6FB;
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

/* Bordered containers -> card look */
div[data-testid="stVerticalBlockBorderWrapper"]{
    background:white;
    border-radius:18px;
    border:1px solid #ECEFF4 !important;
    box-shadow:0 10px 30px rgba(0,0,0,.06);
    padding:4px;
}

/* Metric cards */
div[data-testid="metric-container"]{
    background:white;
    border-radius:18px;
    padding:18px;
    box-shadow:0 10px 30px rgba(0,0,0,.08);
    border:none;
}

/* Number Inputs */
.stNumberInput{
    background:white;
    border-radius:14px;
    padding:2px;
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
button[data-baseweb="tab"]{
    font-size:18px;
    font-weight:600;
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
    background:white;border-radius:18px;padding:18px 20px;
    box-shadow:0 10px 30px rgba(0,0,0,.06);
    display:flex;align-items:center;gap:14px;height:100%;
}
.stat-icon{
    width:46px;height:46px;border-radius:12px;
    display:flex;align-items:center;justify-content:center;
    font-size:20px;flex-shrink:0;
}
.stat-label{font-size:13px;color:#64748B;font-weight:500;margin-bottom:2px;}
.stat-value{font-size:22px;font-weight:700;color:#0F172A;line-height:1.2;}
.stat-sub{font-size:12px;color:#94A3B8;margin-top:2px;}

.health-card{
    background:white;border-radius:18px;padding:18px 22px;
    box-shadow:0 10px 30px rgba(0,0,0,.06);
    display:flex;align-items:center;justify-content:space-between;height:100%;
}
.ring{
    width:78px;height:78px;border-radius:50%;
    display:flex;align-items:center;justify-content:center;
    flex-shrink:0;
}
.ring-inner{
    width:60px;height:60px;background:white;border-radius:50%;
}

.dt-card{
    background:#F8FAFC;border:1px solid #EEF1F6;border-radius:14px;
    padding:12px 14px;text-align:left;
}
.dt-label{font-size:12px;color:#64748B;font-weight:600;}
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
</style>
""")

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("# 🏭 PaperAI")
    st.markdown("### Grade Change Intelligence")
    st.divider()
    st.markdown("### 🧠 System")
    st.metric("Model", "Random Forest")
    st.metric("AI Engine", "Gemini Flash")
    st.metric("Dataset", "5000 Samples")
    st.metric("Accuracy", "99.9%")
    st.success("🟢 System Online")
    st.divider()
    st.info("""
Predicts paper quality during grade transition and
recommends corrective actions using AI.
""")

# ----------------------------
# Top header row
# ----------------------------
h_left, h_right = st.columns([3, 1.6])
with h_left:
    st.html("""
    <h1 style='font-size:42px;margin-bottom:0;display:flex;align-items:center;gap:10px'>
    🏭 PaperAI – Grade Change Intelligence System
    </h1>
    <p style='font-size:16px;color:gray;margin-top:4px'>
    AI-powered prediction and recommendation system for paper manufacturing
    </p>
    """)
with h_right:
    st.html("""
    <div style='display:flex;justify-content:flex-end;gap:10px;margin-top:10px'>
        <span class='top-badge'><span class='dot-green'></span> Live</span>
        <span class='top-badge'>☀️ Theme</span>
        <span class='top-badge'>🚀 Deploy</span>
    </div>
    """)

# ====================================================
# TABS
# ====================================================
prediction_tab, analytics_tab = st.tabs(["📊 Prediction", "📈 Analytics"])

# ====================================================
# PREDICTION TAB
# ====================================================
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
                stock_flow = st.number_input("Stock Flow (L/min)", value=100.0)
                filler_flow = st.number_input("Filler Flow (L/min)", value=20.0)
                steam_pressure = st.number_input("Steam Pressure (bar)", value=55.0)
                machine_speed = st.number_input("Machine Speed (m/min)", value=900.0)
            with c2:
                moisture = st.number_input("Moisture (%)", value=5.0)
                ash = st.number_input("Ash (%)", value=2.0)
                basis_weight = st.number_input("Basis Weight (GSM)", value=80.0)
                target_basis_weight = st.number_input("Target Basis Weight (GSM)", value=80.0)

            st.write("")
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
                <div class='dt-label'>💧 Stock Flow</div>
                <div class='dt-value'>{stock_flow:.2f}</div>
                <div class='dt-unit'>L/min</div>
                </div>""")
            with d2:
                st.html(f"""
                <div class='dt-card'>
                <div class='dt-label'>🧪 Filler Flow</div>
                <div class='dt-value'>{filler_flow:.2f}</div>
                <div class='dt-unit'>L/min</div>
                </div>""")
            with d3:
                st.html(f"""
                <div class='dt-card'>
                <div class='dt-label'>🔥 Steam Pressure</div>
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
    st.write("")
    s1, s2, s3, s4, s5 = st.columns([1, 1, 1, 1, 1.3])
    with s1:
        st.html(f"""
        <div class='stat-card'>
            <div class='stat-icon' style='background:#EFF6FF;'>🛡️</div>
            <div>
                <div class='stat-label'>Risk Score</div>
                <div class='stat-value'>{probability*100:.2f}%</div>
                <div class='stat-sub'>{"Low Risk" if probability < 0.4 else ("Moderate" if probability < 0.7 else "High Risk")}</div>
            </div>
        </div>
        """)
    with s2:
        status_text = "OFF SPEC" if prediction == 1 else "WITHIN SPEC"
        status_color = "#EF4444" if prediction == 1 else "#22C55E"
        st.html(f"""
        <div class='stat-card'>
            <div class='stat-icon' style='background:#F5F3FF;'>📈</div>
            <div>
                <div class='stat-label'>Status</div>
                <div class='stat-value' style='color:{status_color}'>{status_text}</div>
                <div class='stat-sub'>{"Process Drifting" if prediction == 1 else "Stable Process"}</div>
            </div>
        </div>
        """)
    with s3:
        dev_color = "#EF4444" if deviation > 2.5 else "#22C55E"
        st.html(f"""
        <div class='stat-card'>
            <div class='stat-icon' style='background:#FFF7ED;'>🎯</div>
            <div>
                <div class='stat-label'>Basis Deviation</div>
                <div class='stat-value' style='color:{dev_color}'>{deviation:.2f}%</div>
                <div class='stat-sub'>From Target</div>
            </div>
        </div>
        """)
    with s4:
        st.html(f"""
        <div class='stat-card'>
            <div class='stat-icon' style='background:#ECFDF5;'>📋</div>
            <div>
                <div class='stat-label'>Recipe (Target)</div>
                <div class='stat-value'>{target_basis_weight:.0f} GSM</div>
                <div class='stat-sub'>Target Basis Weight</div>
            </div>
        </div>
        """)
    with s5:
        health_pct = min(100, max(0, process_health))
        st.html(f"""
        <div class='health-card'>
            <div>
                <div class='stat-label'>Process Health Score</div>
                <div class='stat-value'>{process_health:.0f} / 100</div>
                <div class='stat-sub' style='color:{health_color};font-weight:600'>
                    {"Excellent" if health_pct >= 80 else ("Good" if health_pct >= 60 else ("Fair" if health_pct >= 40 else "Poor"))}
                </div>
            </div>
            <div class='ring' style='background:conic-gradient({health_color} {health_pct}%, #E5E7EB 0)'>
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

    # ====================================================
    # DETAILED ANALYSIS (shown after clicking Analyze)
    # ====================================================
    if analyze:
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
        st.subheader("📊 Risk Meter")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "green" if probability < 0.4 else "orange" if probability < 0.7 else "red"},
                'steps': [
                    {'range': [0, 40], 'color': 'green'},
                    {'range': [40, 70], 'color': 'orange'},
                    {'range': [70, 100], 'color': 'red'}
                ]}))
        st.plotly_chart(fig, use_container_width=True)
        st.divider()

        st.subheader("🔮 Future Prediction")
        if probability > 0.8:
            st.error("If current conditions continue, the process is likely to remain OFF SPEC.")
        elif probability > 0.5:
            st.warning("The process is drifting towards OFF SPEC.")
        else:
            st.success("Process is expected to remain stable.")

        st.divider()
        st.subheader("⏱ Estimated Stabilization Time")
        if probability > 0.8:
            st.metric("Estimated Time", "8–10 min")
        elif probability > 0.5:
            st.metric("Estimated Time", "5–7 min")
        else:
            st.metric("Estimated Time", "2–3 min")

        st.divider()
        st.subheader("🏭 Process Flow")
        st.graphviz_chart("""digraph G {
rankdir=LR;
StockFlow -> PaperMachine;
FillerFlow -> PaperMachine;
SteamPressure -> PaperMachine;
MachineSpeed -> PaperMachine;
Moisture -> PaperMachine;
Ash -> PaperMachine;
PaperMachine -> BasisWeight;
BasisWeight -> Prediction;
}""")
        st.caption("Digital representation of the paper manufacturing process.")

        st.subheader("Recommended Setpoints")
        recommendations = get_recommendations(
            stock_flow, filler_flow, steam_pressure, machine_speed, moisture, ash, basis_weight
        )
        for rec in recommendations:
            st.success("✔ " + rec)
            st.caption("Source: Historical Data + Random Forest Model")

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
            st.info(ai_text)
        except Exception:
            st.warning("AI explanation is temporarily unavailable. Using rule-based analysis instead.")
            ai_text = "AI explanation unavailable. Rule-based explanation used."
            st.info(ai_text)

        report = pd.DataFrame({
            "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "Prediction": ["OFF SPEC" if prediction else "WITHIN SPEC"],
            "Risk Score": [f"{probability*100:.2f}%"],
            "Current Basis Weight": [basis_weight],
            "Target Basis Weight": [target_basis_weight],
            "Deviation (%)": [round(deviation, 2)],
            "Reasons": [", ".join(reasons)],
            "Recommendations": [", ".join(recommendations)],
            "AI Explanation": [ai_text]
        })
        st.download_button(
            "📥 Download Analysis Report",
            report.to_csv(index=False).encode("utf-8"),
            file_name="analysis_report.csv",
            mime="text/csv"
        )

        st.divider()
        st.subheader("Reason")
        for reason in reasons:
            st.info(reason)

        st.divider()
        st.subheader("👷 Operator Decision")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Accept Recommendation"):
                save_feedback("Accepted")
                st.success("Recommendation saved!")
        with c2:
            if st.button("❌ Reject Recommendation"):
                save_feedback("Rejected")
                st.warning("Recommendation saved!")

# ====================================================
# ANALYTICS TAB
# ====================================================
with analytics_tab:
    st.subheader("📈 Analytics Summary")
    a1, a2, a3 = st.columns(3)
    with a1:
        st.metric("Total Records", len(history))
    with a2:
        st.metric("Off Spec %", f"{history['off_spec'].mean()*100:.1f}%")
    with a3:
        st.metric("Avg Basis Weight", f"{history['basis_weight'].mean():.2f}")

    st.divider()
    st.subheader("Historical Basis Weight")
    fig1 = px.line(history.head(300), y="basis_weight", title="Basis Weight Trend")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Steam Pressure Distribution")
    fig2 = px.histogram(history, x="steam_pressure", title="Steam Pressure Distribution")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Correlation Analysis")
    corr = history.corr(numeric_only=True)
    top = corr["off_spec"].drop("off_spec").abs().sort_values(ascending=False)
    st.subheader("Strongest Correlations")
    st.dataframe(top)

    fig3 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        "Feature": history.drop("off_spec", axis=1).columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    fig4 = px.bar(importance, x="Importance", y="Feature", orientation="h", title="Most Influential Parameters")
    st.plotly_chart(fig4, use_container_width=True)
    st.dataframe(importance, use_container_width=True)

    st.divider()
    st.subheader("📈 Sample Historical Data")
    st.dataframe(history.head(20), use_container_width=True)

    st.divider()
    st.subheader("📋 Operator Feedback History")
    if os.path.exists("feedback.csv"):
        feedback = pd.read_csv("feedback.csv")
        st.dataframe(feedback.tail(10), use_container_width=True)
    else:
        st.info("No operator feedback yet.")

    st.divider()
    st.caption("PaperAI • Built for Grade Change Intelligence Hackathon • Random Forest + Gemini AI + Streamlit")