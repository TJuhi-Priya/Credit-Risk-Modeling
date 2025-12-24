import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Credit Risk Modeling System",
    page_icon="💳",
    layout="centered"
)

# ===================== TITLE =====================
st.markdown("## 💳 Credit Risk Modeling System")
st.divider()

# ===================== GAUGE CHART =====================
def risk_gauge(risk_label, title):
    color = "green" if risk_label == "Low Risk" else "red"
    fig = go.Figure(go.Indicator(
        mode="gauge",
        value=100,
        gauge={
            "axis": {"range": [0, 100], "visible": False},
            "bar": {"color": color},
            "steps": [{"range": [0, 100], "color": color}]
        },
        title={"text": title, "font": {"size": 28}}
    ))
    fig.update_layout(height=300)
    return fig

# ===================== USER INPUT =====================
def user_inputs():
    st.sidebar.header("Applicant Details")
    age = st.sidebar.slider("Age", 18, 75, 30)
    income = st.sidebar.number_input("Annual Income", min_value=1000, value=50000)
    emp_len = st.sidebar.slider("Employment Length (Years)", 0, 40, 5)
    loan_amt = st.sidebar.number_input("Loan Amount", min_value=500, value=10000)
    loan_rate = st.sidebar.slider("Interest Rate (%)", 1.0, 25.0, 10.0)
    home = st.sidebar.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
    intent = st.sidebar.selectbox("Loan Purpose", ["PERSONAL","EDUCATION","MEDICAL","VENTURE","DEBTCONSOLIDATION","HOMEIMPROVEMENT"])
    grade = st.sidebar.selectbox("Loan Grade", list("ABCDEFG"))
    credit_len = st.sidebar.slider("Credit History Length (Years)", 0, 40, 5)
    default_hist = st.sidebar.selectbox("Previous Default?", ["Yes", "No"])

    data = {
        "person_age": age,
        "person_income": income,
        "person_emp_length": emp_len,
        "loan_amnt": loan_amt,
        "loan_int_rate": loan_rate,
        "loan_percent_income": loan_amt / income,
        "cb_person_cred_hist_length": credit_len,
        "cb_person_default_on_file": 1 if default_hist == "Yes" else 0
    }

    for h in ["RENT", "MORTGAGE", "OWN", "OTHER"]:
        data[f"person_home_ownership_{h}"] = 1 if home == h else 0
    for i in ["PERSONAL","EDUCATION","MEDICAL","VENTURE","DEBTCONSOLIDATION","HOMEIMPROVEMENT"]:
        data[f"loan_intent_{i}"] = 1 if intent == i else 0
    for g in list("ABCDEFG"):
        data[f"loan_grade_{g}"] = 1 if grade == g else 0

    return pd.DataFrame(data, index=[0])

input_df = user_inputs()

# ===================== LOAD MODEL =====================
column_order = joblib.load("saved_model/column_order.pkl")
model = joblib.load("saved_model/rfc.pkl")

# ===================== FEATURE ALIGNMENT =====================
for col in column_order:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[column_order]

# ===================== PREDICTION =====================
prediction = model.predict(input_df)
probability = model.predict_proba(input_df)[0][1]

st.subheader("🔍 Risk Prediction")
if prediction[0] == 0:
    st.plotly_chart(risk_gauge("Low Risk", "LOW CREDIT RISK"))
    st.success("Applicant is predicted to be **LOW RISK** for loan default.")
else:
    st.plotly_chart(risk_gauge("High Risk", "HIGH CREDIT RISK"))
    st.error("Applicant is predicted to be **HIGH RISK** for loan default.")
st.write(f"**Probability of Default:** `{probability:.2%}`")

st.divider()

# ===================== LIME EXPLAINABILITY =====================
st.subheader("🧠 Model Explainability (LIME)")

explainer = LimeTabularExplainer(
    training_data=input_df.values,
    feature_names=input_df.columns.tolist(),
    class_names=["Low Risk", "High Risk"],
    mode="classification"
)

exp = explainer.explain_instance(
    data_row=input_df.values[0],
    predict_fn=model.predict_proba,
    num_features=8
)

fig = exp.as_pyplot_figure()
st.pyplot(fig)

# ===================== FOOTER =====================
st.divider()
st.caption(
    "✔ SMOTE for Imbalance | ✔ Random Forest Ensemble | ✔ LIME Explainability | "
    "✔ LendingClub-style Dataset | ✔ Production-ready Deployment"
)
