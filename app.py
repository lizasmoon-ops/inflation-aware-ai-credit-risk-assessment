import streamlit as st
import pandas as pd
import joblib

model = joblib.load("credit_risk_model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("Inflation-Aware Credit Risk Decision Support Tool")

st.write("This tool predicts borrower default risk and tests how risk changes under inflation scenarios.")

age = st.number_input("Age", min_value=18, max_value=100, value=40)
income = st.number_input("Monthly Income", min_value=0.0, value=3000.0)
debt_ratio = st.number_input("Debt Ratio", min_value=0.0, value=0.35)
credit_utilisation = st.number_input("Credit Utilisation", min_value=0.0, value=0.45)
credit_lines = st.number_input("Number of Open Credit Lines and Loans", min_value=0, value=5)
late_90 = st.number_input("Number of Times 90 Days Late", min_value=0, value=0)
dependents = st.number_input("Number of Dependents", min_value=0, value=0)

def risk_category(probability):
    if probability < 0.05:
        return "Low Risk"
    elif probability < 0.10:
        return "Medium Risk"
    elif probability < 0.20:
        return "High Risk"
    else:
        return "Very High Risk"

def recommendation(probability):
    if probability < 0.05:
        return "Approve"
    elif probability < 0.10:
        return "Approve with Monitoring"
    elif probability < 0.20:
        return "Manual Review Required"
    else:
        return "Reject or Require Strong Mitigation"

def create_borrower_data(income, debt_ratio, credit_utilisation):
    borrower = pd.DataFrame(columns=feature_names)
    borrower.loc[0] = 0

    borrower["RevolvingUtilizationOfUnsecuredLines"] = credit_utilisation
    borrower["age"] = age
    borrower["DebtRatio"] = debt_ratio
    borrower["MonthlyIncome"] = income
    borrower["NumberOfOpenCreditLinesAndLoans"] = credit_lines
    borrower["NumberOfTimes90DaysLate"] = late_90
    borrower["NumberOfDependents"] = dependents

    if "DebtPerDependent" in borrower.columns:
        borrower["DebtPerDependent"] = borrower["DebtRatio"] / (borrower["NumberOfDependents"] + 1)

    if "CreditLinesPerAge" in borrower.columns:
        borrower["CreditLinesPerAge"] = borrower["NumberOfOpenCreditLinesAndLoans"] / borrower["age"]

    if "IncomeToDebt" in borrower.columns:
        borrower["IncomeToDebt"] = borrower["MonthlyIncome"] / (borrower["DebtRatio"] + 1)

    return borrower[feature_names]

if st.button("Assess Risk"):

    normal = create_borrower_data(income, debt_ratio, credit_utilisation)
    normal_prob = model.predict_proba(normal)[0][1]

    mild = create_borrower_data(income * 0.95, debt_ratio * 1.10, credit_utilisation * 1.05)
    mild_prob = model.predict_proba(mild)[0][1]

    moderate = create_borrower_data(income * 0.90, debt_ratio * 1.20, credit_utilisation * 1.15)
    moderate_prob = model.predict_proba(moderate)[0][1]

    severe = create_borrower_data(income * 0.85, debt_ratio * 1.30, credit_utilisation * 1.25)
    severe_prob = model.predict_proba(severe)[0][1]

    st.subheader("Normal Credit Risk Assessment")
    st.write(f"Default Probability: {normal_prob:.2%}")
    st.write(f"Risk Category: {risk_category(normal_prob)}")
    st.write(f"Recommendation: {recommendation(normal_prob)}")

    st.subheader("Inflation Stress Testing Results")

    results = pd.DataFrame({
        "Scenario": ["Normal", "Mild", "Moderate", "Severe"],
        "Default Probability": [normal_prob, mild_prob, moderate_prob, severe_prob],
        "Risk Category": [
            risk_category(normal_prob),
            risk_category(mild_prob),
            risk_category(moderate_prob),
            risk_category(severe_prob)
        ]
    })

    st.dataframe(results)
    st.bar_chart(results.set_index("Scenario")["Default Probability"])