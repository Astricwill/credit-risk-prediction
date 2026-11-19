import streamlit as st
import pandas as pd
import joblib
import shap
import plotly.graph_objects as go

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")
model = joblib.load("src/models/credit_risk_model.pkl")

st.title("Credit Risk & Loan Default Predictor")
st.write("Enter applicant details to estimate default probability.")
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", 18, 90, 35)
    annual_income = st.number_input("Annual income ($)", 0, 1_000_000, 55000)
    employment_years = st.number_input("Years employed", 0, 50, 5)
with col2:
    loan_amount = st.number_input("Loan amount ($)", 500, 200000, 15000)
    loan_term_months = st.selectbox("Loan term (months)", [12, 24, 36, 48, 60])
    credit_score = st.slider("Credit score", 300, 850, 680)
with col3:
    existing_debt = st.number_input("Existing debt ($)", 0, 500000, 8000)
    num_late_payments_2yr = st.number_input("Late payments (2yr)", 0, 20, 0)
    num_open_credit_lines = st.number_input("Open credit lines", 0, 30, 4)
    home_ownership = st.selectbox("Home ownership", ["RENT", "OWN", "MORTGAGE"])
    loan_purpose = st.selectbox(
    "Loan purpose",
    ["debt_consolidation", "home_improvement", "small_business", "car", "medical"],
    )

if st.button("Assess Risk"):
    applicant = pd.DataFrame([{
    "age": age, "annual_income": annual_income, "employment_years": employment_years,
    "loan_amount": loan_amount, "loan_term_months": loan_term_months,
    "credit_score": credit_score, "existing_debt": existing_debt,
    "debt_to_income": existing_debt / max(annual_income, 1),
    "loan_to_income": loan_amount / max(annual_income, 1),
    "num_late_payments_2yr": num_late_payments_2yr,
    "num_open_credit_lines": num_open_credit_lines,
    "home_ownership": home_ownership, "loan_purpose": loan_purpose,
    }])

    proba = model.predict_proba(applicant)[0, 1]

    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=proba * 100,
        title={"text": "Default Probability (%)"},
        gauge={"axis": {"range": [0, 100]},
        "steps": [{"range": [0, 20], "color": "lightgreen"},
        {"range": [20, 50], "color": "gold"},
        {"range": [50, 100], "color": "lightcoral"}]},
    ))
    st.plotly_chart(fig, use_container_width=True)
    
    if proba < 0.2:
        st.success("Low risk — likely approve.")
    elif proba < 0.5:
        st.warning("Medium risk — manual review recommended.")
    else:
        st.error("High risk — likely decline.")
