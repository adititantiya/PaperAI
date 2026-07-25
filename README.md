# 🏭 PaperAI – Grade Change Intelligence for Paper Manufacturing

PaperAI is an AI-powered Grade Change Intelligence platform built for the Honeywell Hackathon that predicts OFF-SPEC paper production during grade transitions, recommends corrective machine setpoints, and explains every recommendation using historical production data and Google Gemini.

The system combines **Machine Learning, Explainable AI, historical process analytics, and operator feedback** to help operators reduce stabilization time, minimize production losses, and make informed decisions during paper manufacturing.

---

# ✨ Key Highlights

- 🤖 Random Forest based OFF-SPEC prediction
- 📈 Real-time process risk assessment
- 🎯 Intelligent corrective setpoint recommendations
- 📊 Historical trajectory comparison with successful production runs
- 🧠 Explainable AI powered by Google Gemini
- ⭐ AI recommendation confidence scoring
- 📉 Correlation & feature importance analytics
- 👷 Operator feedback tracking
- 📄 Exportable CSV & PDF production reports
- 🏭 Interactive Digital Twin visualization

---

# 📌 Problem Statement

During paper grade transitions, multiple process variables such as stock flow, filler flow, steam pressure, machine speed, moisture, ash, and basis weight change simultaneously.

These transitions often produce OFF-SPEC paper before the process stabilizes, resulting in:

- Material loss
- Increased stabilization time
- Higher production cost
- Heavy operator dependency
- Lack of explainable decision support

Traditional automation follows predefined trajectories but does not learn from historical production data.

PaperAI introduces an intelligent prediction and recommendation layer that learns from historical successful production runs and assists operators in making proactive decisions.

---

# 🚀 Solution Overview

PaperAI continuously analyzes current process conditions and predicts whether the paper machine is likely to produce OFF-SPEC paper during a grade transition.

The platform provides:

- Real-time OFF-SPEC prediction
- Risk score estimation
- Historical trajectory comparison
- Intelligent process recommendations
- AI-generated explanation
- Analytics dashboard
- Operator feedback collection
- Exportable reports

---

# ⚙ Core Features

## 1. OFF-SPEC Prediction

A trained **Random Forest Classifier** predicts whether the current operating conditions will produce paper within specification.

### Input Parameters

- Stock Flow
- Filler Flow
- Steam Pressure
- Machine Speed
- Moisture
- Ash
- Basis Weight
- Target Basis Weight
- Basis Weight Deviation

### Output

- WITHIN SPEC
- OFF SPEC
- Risk Score (%)

---

## 2. Intelligent Recommendation Engine

When abnormal operating conditions are detected, PaperAI recommends corrective machine settings based on historical successful production runs.

Examples include:

- Reduce Steam Pressure
- Adjust Machine Speed
- Reduce Moisture
- Correct Basis Weight
- Increase Stock Flow

These recommendations help operators stabilize the process before quality limits are exceeded.

---

## 3. Historical Trajectory Comparison

Instead of only suggesting new values, PaperAI compares the current process with historical successful production runs.

Operators can instantly view:

- Current Value
- Historical Best Value
- Required Adjustment

This enables data-driven decision making using historical plant knowledge.

---

## 4. AI Recommendation Confidence

Every recommendation includes a confidence score derived from historical production data and engineering rules.

Example:

| Parameter | Confidence |
|------------|------------|
| Steam Pressure | 96% |
| Machine Speed | 93% |
| Moisture | 91% |
| Basis Weight | 98% |
| Stock Flow | 90% |

---

## 5. Basis Weight Monitoring

The application continuously compares:

- Current Basis Weight
- Target Basis Weight

and calculates:

- Percentage Deviation
- ±2.5% Specification Check

This directly addresses the primary hackathon objective of predicting basis weight deviation during grade transitions.

---

## 6. Future Process Prediction

Based on the predicted probability, PaperAI forecasts whether the process is expected to:

- Remain Stable
- Drift Toward OFF-SPEC
- Remain OFF-SPEC

along with estimated stabilization time after corrective actions.

---

## 7. Explainable AI

Google Gemini generates natural language explanations describing:

- Why the prediction was made
- Important influencing parameters
- Corrective recommendations
- Expected impact

If Gemini is unavailable, the application automatically switches to a rule-based explanation to ensure uninterrupted operation.

---

## 8. Interactive Analytics Dashboard

The analytics module provides comprehensive production insights including:

- Historical Basis Weight Trend
- Steam Pressure Distribution
- Correlation Heatmap
- Feature Importance Analysis
- Steam Pressure vs Basis Weight Analysis
- Historical Trajectory Comparison
- Operator Feedback Analytics
- Model Performance Metrics
- Sample Historical Dataset

---

## 9. Digital Twin Visualization

PaperAI includes an interactive Digital Twin representing the paper manufacturing process and showing relationships between:

- Stock Flow
- Steam Pressure
- Machine Speed
- Moisture
- Basis Weight
- Quality Prediction

This helps operators understand process dependencies during grade transitions.

---

## 10. Operator Feedback

Operators can evaluate recommendations by selecting:

- ✅ Accept Recommendation
- ❌ Reject Recommendation

All responses are logged for future analysis and recommendation quality assessment.

---

## 11. Report Generation

Every prediction can be exported as:

- CSV Production Report
- PDF Executive Report

Reports include:

- Prediction
- Risk Score
- Current Parameters
- Recommended Settings
- Basis Weight Deviation
- Historical Comparison
- AI Explanation
- Operator Decision
- Timestamp

---

# 🧠 Machine Learning

### Algorithm

Random Forest Classifier

### Training Dataset

5,000 simulated historical production records representing successful and failed grade transition scenarios.

### Input Features

- Stock Flow
- Filler Flow
- Steam Pressure
- Machine Speed
- Moisture
- Ash
- Basis Weight
- Target Basis Weight
- Basis Weight Deviation

### Output Classes

- WITHIN SPEC
- OFF SPEC

---

# 🏗 System Architecture

```
Current Process Parameters
            │
            ▼
 Random Forest Prediction
            │
            ▼
 OFF-SPEC Risk Assessment
            │
            ▼
 Historical Run Comparison
            │
            ▼
 Recommendation Engine
            │
            ▼
 Google Gemini Explanation
            │
            ▼
 Operator Decision
            │
            ▼
 Feedback Logging
            │
            ▼
 CSV / PDF Report Generation
```

---

# 💻 Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Machine Learning | Scikit-learn Random Forest |
| Explainable AI | Google Gemini Flash |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Model Storage | Joblib |
| Report Generation | ReportLab |
| Programming Language | Python |

---

# 📂 Project Structure

```
PaperAI/
│
├── app.py
├── train.py
├── generate_data.py
├── recommendation.py
├── gemini_helper.py
├── feedback.py
│
├── data/
│   └── historical_data.csv
│
├── models/
│   ├── random_forest.pkl
│   └── confusion_matrix.png
│
├── feedback.csv
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/adititantiya/PaperAI.git

cd PaperAI
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 🔄 Workflow

```
Input Process Parameters
            │
            ▼
Predict OFF-SPEC Probability
            │
            ▼
Evaluate Basis Weight Deviation
            │
            ▼
Compare With Historical Successful Runs
            │
            ▼
Generate Corrective Recommendations
            │
            ▼
Explain Using Google Gemini
            │
            ▼
Operator Accepts / Rejects Recommendation
            │
            ▼
Generate PDF / CSV Report
```

---

# 📊 Alignment with Honeywell Hackathon Objectives

| Requirement | Status |
|-------------|--------|
| Predict basis weight deviation | ✅ |
| Predict OFF-SPEC during grade change | ✅ |
| Recommend corrective setpoints | ✅ |
| Explain prediction rationale | ✅ |
| Use historical successful runs | ✅ |
| Historical trajectory comparison | ✅ |
| Correlation & analytics dashboard | ✅ |
| Digital Twin visualization | ✅ |
| Operator accept/reject workflow | ✅ |
| Store operator responses | ✅ |
| Downloadable reports | ✅ |

---

# 🔮 Future Scope

- Integration with Honeywell QCS / PLC / DCS
- OPC-UA / MQTT based real-time data streaming
- Time-series prediction using LSTM/XGBoost
- Adaptive learning from operator feedback
- Multi-grade recipe optimization
- Reinforcement learning based control
- Cloud deployment with live dashboards

---

# 👨‍💻 Authors

Developed for the **Honeywell Grade Change Intelligence Hackathon**.

PaperAI demonstrates how **Machine Learning, Explainable AI, historical production analytics, and operator feedback** can reduce OFF-SPEC production, minimize stabilization time, and improve decision-making during paper grade transitions.

---

## 📜 License

This project was developed for the **Honeywell Grade Change Intelligence Hackathon** and is intended for educational and demonstration purposes.