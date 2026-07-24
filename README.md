# 🏭 PaperAI – Grade Change Intelligence for Paper Manufacturing

An AI-powered Grade Change Intelligence System developed for the Honeywell Hackathon to predict paper quality deviations during grade transitions and recommend corrective actions before off-spec production occurs.

---

## 📌 Problem Statement

During paper grade transitions, multiple process variables such as stock flow, filler flow, steam pressure, machine speed, moisture, ash, and basis weight change simultaneously. These transitions often produce off-spec paper while the process stabilizes, resulting in:

- Material loss
- Increased stabilization time
- Higher production cost
- Operator dependency
- Lack of explainable decision support

Traditional automation executes predefined trajectories but does not learn from historical operating data.

PaperAI adds an intelligent prediction and recommendation layer on top of existing process control.

---

# 🚀 Solution Overview

PaperAI predicts whether the paper manufacturing process is likely to go **OFF SPEC** during a grade change and provides:

- Real-time quality prediction
- Risk score estimation
- Recommended process setpoints
- Basis weight deviation analysis
- Explainable AI reasoning (Gemini)
- Historical correlation dashboard
- Operator feedback logging
- Downloadable analysis reports

---

# Features

## 1. Off-Spec Prediction

Uses a trained Random Forest classifier to determine whether the current operating conditions are likely to produce off-spec paper.

Input Parameters

- Stock Flow
- Filler Flow
- Steam Pressure
- Machine Speed
- Moisture
- Ash
- Basis Weight
- Target Basis Weight

Output

- WITHIN SPEC
- OFF SPEC
- Risk Score (%)

---

## 2. Corrective Setpoint Recommendation

When abnormal operating conditions are detected, PaperAI recommends corrective actions such as:

- Reduce steam pressure
- Adjust machine speed
- Reduce moisture
- Correct basis weight deviation

These recommendations help operators stabilize the process before quality limits are exceeded.

---

## 3. Basis Weight Monitoring

The application continuously compares

Current Basis Weight

with

Target Basis Weight

and computes

- Percentage deviation
- ±2.5% tolerance check

This directly addresses the primary challenge of predicting basis weight deviation during grade transitions.

---

## 4. Future Process Prediction

Based on the predicted probability, PaperAI forecasts whether the process is expected to

- remain stable
- drift towards OFF SPEC
- remain OFF SPEC

allowing proactive operator intervention.

---

## 5. Explainable AI

Google Gemini generates natural-language explanations describing

- why the prediction was made
- important influencing parameters
- recommended corrective actions

If Gemini is unavailable, the system automatically switches to a rule-based explanation so the application remains functional.

---

## 6. Analytics Dashboard

The dashboard provides historical process insights including

- Basis Weight Trend
- Steam Pressure Distribution
- Correlation Heatmap
- Feature Importance
- Sample Historical Dataset
- Operator Feedback History

This enables engineers to identify relationships between process variables.

---

## 7. Operator Feedback

Operators can

- Accept Recommendation
- Reject Recommendation

All responses are stored in a feedback log for future evaluation of recommendation quality.

---

## 8. Analysis Report

Each prediction can be exported as a CSV report containing

- Timestamp
- Prediction
- Risk Score
- Current Basis Weight
- Target Basis Weight
- Basis Weight Deviation
- Reasons
- Recommendations
- AI Explanation

---

# Machine Learning

Algorithm

- Random Forest Classifier

Training Dataset

- 5000 historical process records

Input Features

- Stock Flow
- Filler Flow
- Steam Pressure
- Machine Speed
- Moisture
- Ash
- Basis Weight
- Target Basis Weight
- Basis Weight Deviation

Prediction

- Binary Classification
    - Within Spec
    - Off Spec

---

# Technology Stack

Frontend

- Streamlit

Machine Learning

- Scikit-learn
- Random Forest

AI Explanation

- Google Gemini Flash

Visualization

- Plotly
- Pandas

Model Storage

- Joblib

Language

- Python

---

# Project Structure

```
PaperAI
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
│   └── feature_importance.png
│
├── feedback.csv
├── requirements.txt
└── README.md
```

---

# Installation

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

# Expected Workflow

1. Enter current process parameters.

2. Click **Analyze Process**.

3. The model predicts process quality.

4. Risk score is calculated.

5. Basis weight deviation is evaluated.

6. Recommended setpoints are generated.

7. AI explains the prediction.

8. Operator accepts or rejects recommendations.

9. Feedback is stored.

10. Download the analysis report.

---

# Alignment with Hackathon Objectives

| Requirement | Status |
|-------------|--------|
| Predict basis weight deviation | ✅ |
| Predict off-spec during grade change | ✅ |
| Recommend corrective setpoints | ✅ |
| Explain prediction | ✅ |
| Show historical correlations | ✅ |
| Dashboard with analytics | ✅ |
| Operator accept/reject suggestions | ✅ |
| Record operator responses | ✅ |
| Export analysis report | ✅ |

---

# Future Improvements

- Integration with live Honeywell QCS/PLC data
- Time-series prediction using LSTM/XGBoost
- Digital Twin visualization
- Multi-grade recipe optimization
- Real-time sensor streaming
- Adaptive learning from operator feedback
- Production deployment with REST APIs

---

# Authors

Developed for the **Honeywell Grade Change Intelligence Hackathon**.

PaperAI demonstrates how machine learning and explainable AI can support operators in reducing off-spec production, minimizing stabilization time, and improving decision-making during paper grade transitions.