import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from gemini_helper import explain
from feedback import save_feedback
from datetime import datetime
import plotly.graph_objects as go
import os


from recommendation import get_recommendations

# ----------------------------
# Load Model & Data
# ----------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

model = joblib.load("models/random_forest.pkl")
history = pd.read_csv("data/historical_data.csv")

# ----------------------------
# Page
# ----------------------------

st.set_page_config(
    page_title="Paper Grade Intelligence",
    page_icon="📄",
    layout="wide"
)
st.markdown("""
<style>

/* Background */
.stApp{
    background:#F3F6FB;
    font-family:'Poppins',sans-serif;
}

/* Entire App Font */
html, body, [class*="css"]{
    font-family:'Poppins',sans-serif;
}

/* Main Container */
.block-container{
    padding-top:1.5rem;
    padding-left:2rem;
    padding-right:2rem;
}

/* Sidebar */

[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#08152C,#142A4F);
}

[data-testid="stSidebar"] *{
    color:white;
}

/* Metric Cards */

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

    padding:10px;

    border:1px solid #E7EAF0;

}

/* Buttons */

.stButton>button{

    background:linear-gradient(90deg,#2563EB,#4F46E5);

    color:white;

    border:none;

    border-radius:14px;

    height:55px;

    font-size:18px;

    font-weight:600;

    transition:.25s;

}

.stButton>button:hover{

    transform:translateY(-2px);

    box-shadow:0 10px 20px rgba(79,70,229,.35);

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

h1{

    font-weight:700;

}

h2{

    color:#1E3A8A;

}

h3{

    color:#1E3A8A;

}

</style>
""", unsafe_allow_html=True)
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

st.markdown("""
<h1 style='font-size:48px;margin-bottom:0'>
🏭 PaperAI
</h1>

<h3 style='margin-top:0;color:#555'>
Grade Change Intelligence System
</h3>

<p style='font-size:18px;color:gray'>
AI-powered prediction and recommendation system for paper manufacturing
</p>
""", unsafe_allow_html=True)

# ----------------------------
# Tabs
# ----------------------------

prediction_tab, analytics_tab = st.tabs([
    "📊 Prediction",
    "📈 Analytics"
])

# ====================================================
# PREDICTION TAB
# ====================================================

with prediction_tab:

    col1, col2 = st.columns(2)

    with col1:
        stock_flow = st.number_input("Stock Flow", value=100.0)
        filler_flow = st.number_input("Filler Flow", value=20.0)
        steam_pressure = st.number_input("Steam Pressure", value=55.0)
        machine_speed = st.number_input("Machine Speed", value=900.0)

    with col2:
        moisture = st.number_input("Moisture", value=5.0)
        ash = st.number_input("Ash", value=2.0)
        basis_weight = st.number_input("Basis Weight", value=80.0)
        target_basis_weight = st.number_input("Target Basis Weight",value=80.0)

    st.divider()

    col_run, col_reset = st.columns(2)
    with col_run:
        analyze = st.button("🚀 Analyze Process", use_container_width=True)
    with col_reset:
        reset = st.button("🔄 Reset", use_container_width=True)
    if reset:
        st.rerun()
    if analyze:
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

        st.subheader("📊 Process Summary")

        c1, c2, c3 = st.columns(3)

        with c1:
            if prediction == 1:
                st.error("⚠ OFF SPEC")
            else:
                st.success("✅ WITHIN SPEC")
            k1, k2 = st.columns(2)
            with k1:
                st.metric(
        "🎯 Target BW",
        f"{target_basis_weight:.1f} GSM"
    )
            with k2:
                st.metric(
        "📄 Current BW",
        f"{basis_weight:.1f} GSM"
    )

        with c2:
            st.metric("Risk Score", f"{probability*100:.2f}%")
            
            st.metric("Basis Weight Deviation",f"{deviation:.2f}%")
            if deviation > 2.5:
                st.error("⚠ Basis Weight exceeds ±2.5% tolerance")
            else:
                st.success("✅ Basis Weight within tolerance")
        with c3:
            if probability > 0.8:
                st.metric("Status", "🔴 Critical")
            elif probability > 0.5:
                st.metric("Status", "🟠 Warning")
            else:
                st.metric("Status", "🟢 Stable")
            process_health = max(0, 100 - probability * 100)
            st.metric( "Process Health",f"{process_health:.1f}/100")

        st.divider()
        st.subheader("📊 Risk Meter")
        fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=probability*100,
    title={'text':"Risk Score"},
    gauge={
        'axis':{'range':[0,100]},
        'bar': {
    'color':
        "green" if probability < 0.4
        else "orange" if probability < 0.7
        else "red"
},
        'steps':[
            {'range':[0,40],'color':'green'},
            {'range':[40,70],'color':'orange'},
            {'range':[70,100],'color':'red'}
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
            stock_flow,
            filler_flow,
            steam_pressure,
            machine_speed,
            moisture,
            ash,
            basis_weight
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
        probability,
        recommendations,
        values)
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

        st.divider()


# ====================================================
# ANALYTICS TAB
# ====================================================

with analytics_tab:
    st.subheader("📈 Analytics Summary")
    a1, a2, a3 = st.columns(3)
    with a1:
        st.metric("Total Records", len(history))
    with a2:
        st.metric(
        "Off Spec %",
        f"{history['off_spec'].mean()*100:.1f}%"
    )
    with a3:
            st.metric(
        "Avg Basis Weight",
        f"{history['basis_weight'].mean():.2f}"
    )
    st.divider()
    st.subheader("Historical Basis Weight")

    fig1 = px.line(
        history.head(300),
        y="basis_weight",
        title="Basis Weight Trend"
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Steam Pressure Distribution")

    fig2 = px.histogram(
        history,
        x="steam_pressure",
        title="Steam Pressure Distribution"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Correlation Analysis")
    

    corr = history.corr(numeric_only=True)
    top = (
    corr["off_spec"]
    .drop("off_spec")
    .abs()
    .sort_values(ascending=False))
    st.subheader("Strongest Correlations")
    st.dataframe(top)

    fig3 = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r"
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Feature Importance")

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
        title="Most Influential Parameters"
    )

    st.plotly_chart(fig4, use_container_width=True)
    st.dataframe(importance,use_container_width=True)
    st.divider()
    st.subheader("📈 Sample Historical Data")
    st.dataframe(
    history.head(20),
    use_container_width=True
)


    st.divider()
    st.subheader("📋 Operator Feedback History")
    if os.path.exists("feedback.csv"):
        feedback = pd.read_csv("feedback.csv")
        st.dataframe(
        feedback.tail(10),
        use_container_width=True
    )
    else:
        st.info("No operator feedback yet.")
    st.divider()
    st.caption(
    "PaperAI • Built for Grade Change Intelligence Hackathon • Random Forest + Gemini AI + Streamlit"
)