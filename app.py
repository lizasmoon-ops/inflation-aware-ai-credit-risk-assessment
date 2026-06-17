import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Inflation-Aware Credit Risk Tool",
    page_icon="🏦",
    layout="wide"
)

# -------------------------------
# Load Model
# -------------------------------
model = joblib.load("credit_risk_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# -------------------------------
# Custom Styling
# -------------------------------
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.header-box {
    background: linear-gradient(90deg, #0f172a, #1e3a8a);
    padding: 35px;
    border-radius: 18px;
    color: white;
    margin-bottom: 30px;
}

.metric-card {
    background-color: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    text-align: center;
    margin-bottom: 15px;
}

.section-card {
    background-color: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.06);
    margin-bottom: 25px;
}

.low {
    color: #15803d;
    font-weight: bold;
}

.medium {
    color: #ca8a04;
    font-weight: bold;
}

.high {
    color: #ea580c;
    font-weight: bold;
}

.veryhigh {
    color: #dc2626;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Functions
# -------------------------------
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

def risk_class(category):
    if category == "Low Risk":
        return "low"
    elif category == "Medium Risk":
        return "medium"
    elif category == "High Risk":
        return "high"
    else:
        return "veryhigh"

def create_borrower_data(age, income, debt_ratio, credit_utilisation, credit_lines, late_90, dependents):
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

# -------------------------------
# Header
# -------------------------------
st.markdown("""
<div class="header-box">
    <h1>🏦 Inflation-Aware Credit Risk Decision Support Tool</h1>
    <p style="font-size:18px;">
    AI-powered borrower default prediction with inflation stress-testing and lending recommendation support.
    </p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.title("Borrower Profile")
st.sidebar.write("Enter borrower information below.")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=40)
income = st.sidebar.number_input("Monthly Income", min_value=0.0, value=3000.0)
debt_ratio = st.sidebar.number_input("Debt Ratio", min_value=0.0, value=0.35)
credit_utilisation = st.sidebar.number_input("Credit Utilisation", min_value=0.0, value=0.45)
credit_lines = st.sidebar.number_input("Open Credit Lines and Loans", min_value=0, value=5)
late_90 = st.sidebar.number_input("Times 90 Days Late", min_value=0, value=0)
dependents = st.sidebar.number_input("Number of Dependents", min_value=0, value=0)

assess = st.sidebar.button("Run Credit Risk Assessment")

# -------------------------------
# Main Page Before Assessment
# -------------------------------
if not assess:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="section-card">
            <h3>1. Default Prediction</h3>
            <p>The system uses a trained Gradient Boosting model to estimate borrower default probability.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="section-card">
            <h3>2. Inflation Stress Testing</h3>
            <p>The tool simulates mild, moderate and severe inflation scenarios to assess changing borrower risk.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="section-card">
            <h3>3. Lending Recommendation</h3>
            <p>The system assigns a risk category and produces a decision-support recommendation.</p>
        </div>
        """, unsafe_allow_html=True)

    st.info("Use the sidebar to enter borrower details and run the assessment.")

# -------------------------------
# Assessment Results
# -------------------------------
if assess:
    normal = create_borrower_data(age, income, debt_ratio, credit_utilisation, credit_lines, late_90, dependents)
    normal_prob = model.predict_proba(normal)[0][1]

    mild = create_borrower_data(age, income * 0.95, debt_ratio * 1.10, credit_utilisation * 1.05, credit_lines, late_90, dependents)
    mild_prob = model.predict_proba(mild)[0][1]

    moderate = create_borrower_data(age, income * 0.90, debt_ratio * 1.20, credit_utilisation * 1.15, credit_lines, late_90, dependents)
    moderate_prob = model.predict_proba(moderate)[0][1]

    severe = create_borrower_data(age, income * 0.85, debt_ratio * 1.30, credit_utilisation * 1.25, credit_lines, late_90, dependents)
    severe_prob = model.predict_proba(severe)[0][1]

    category = risk_category(normal_prob)
    rec = recommendation(normal_prob)

    st.subheader("Credit Risk Assessment Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Default Probability</h4>
            <h2>{normal_prob:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Risk Category</h4>
            <h2 class="{risk_class(category)}">{category}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Recommendation</h4>
            <h2>{rec}</h2>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Inflation Stress Testing Results")

    results = pd.DataFrame({
        "Scenario": ["Normal", "Mild Inflation", "Moderate Inflation", "Severe Inflation"],
        "Income Adjustment": ["None", "-5%", "-10%", "-15%"],
        "Debt Ratio Adjustment": ["None", "+10%", "+20%", "+30%"],
        "Credit Utilisation Adjustment": ["None", "+5%", "+15%", "+25%"],
        "Default Probability": [normal_prob, mild_prob, moderate_prob, severe_prob],
        "Risk Category": [
            risk_category(normal_prob),
            risk_category(mild_prob),
            risk_category(moderate_prob),
            risk_category(severe_prob)
        ],
        "Recommendation": [
            recommendation(normal_prob),
            recommendation(mild_prob),
            recommendation(moderate_prob),
            recommendation(severe_prob)
        ]
    })

    display_results = results.copy()
    display_results["Default Probability"] = display_results["Default Probability"].apply(lambda x: f"{x:.2%}")

    st.dataframe(display_results, use_container_width=True)

    chart_data = pd.DataFrame({
        "Scenario": ["Normal", "Mild", "Moderate", "Severe"],
        "Default Probability": [normal_prob, mild_prob, moderate_prob, severe_prob]
    })

    st.bar_chart(chart_data.set_index("Scenario"))

    st.subheader("Interpretation")

    if severe_prob > normal_prob:
        increase = severe_prob - normal_prob
        st.warning(
            f"Under severe inflation conditions, the borrower's predicted default probability increases by {increase:.2%}. "
            "This indicates that inflationary pressure can worsen borrower risk and should be considered in lending decisions."
        )
    else:
        st.info(
            "The model does not show a major increase in risk under stress conditions for this borrower profile."
        )

    st.caption(
        "Disclaimer: This application is a dissertation prototype. In a real banking environment, the system would require secure authentication, regulatory validation, explainability controls and private deployment."
    )
