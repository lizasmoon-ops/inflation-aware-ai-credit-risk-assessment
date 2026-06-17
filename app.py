import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Inflation-Aware Credit Risk Tool",
    page_icon="🏦",
    layout="wide"
)

model = joblib.load("credit_risk_model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.markdown("""
<style>
.header {
    background: linear-gradient(90deg, #0f172a, #1e40af);
    padding: 30px;
    border-radius: 15px;
    color: white;
    margin-bottom: 25px;
}
.card {
    background-color: white;
    padding: 22px;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.metric-title {
    font-size: 16px;
    color: #475569;
}
.metric-value {
    font-size: 30px;
    font-weight: bold;
    color: #0f172a;
}
.low {color: #15803d; font-weight: bold;}
.medium {color: #ca8a04; font-weight: bold;}
.high {color: #ea580c; font-weight: bold;}
.veryhigh {color: #dc2626; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

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

st.markdown("""
<div class="header">
<h1>🏦 Inflation-Aware Credit Risk Decision Support Tool</h1>
<p>AI-based borrower default prediction with inflation stress testing and lending recommendation support.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("Borrower Profile")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=40)
income = st.sidebar.number_input("Monthly Income", min_value=0.0, value=3000.0)
debt_ratio = st.sidebar.number_input("Debt Ratio", min_value=0.0, value=0.35)
credit_utilisation = st.sidebar.number_input("Credit Utilisation", min_value=0.0, value=0.45)
credit_lines = st.sidebar.number_input("Open Credit Lines and Loans", min_value=0, value=5)
late_90 = st.sidebar.number_input("Times 90 Days Late", min_value=0, value=0)
dependents = st.sidebar.number_input("Number of Dependents", min_value=0, value=0)

run_assessment = st.sidebar.button("Run Assessment")

if not run_assessment:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
        <h3>1. Credit Risk Prediction</h3>
        <p>The system estimates borrower default probability using a trained Gradient Boosting model.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
        <h3>2. Inflation Stress Testing</h3>
        <p>The tool simulates mild, moderate and severe inflationary pressure on borrower financial conditions.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
        <h3>3. Lending Decision Support</h3>
        <p>The system produces a risk category and recommendation to support lending assessment.</p>
        </div>
        """, unsafe_allow_html=True)

    st.info("Enter borrower details in the sidebar and click Run Assessment.")

if run_assessment:
    normal = create_borrower_data(age, income, debt_ratio, credit_utilisation, credit_lines, late_90, dependents)
    mild = create_borrower_data(age, income * 0.95, debt_ratio * 1.10, credit_utilisation * 1.05, credit_lines, late_90, dependents)
    moderate = create_borrower_data(age, income * 0.90, debt_ratio * 1.20, credit_utilisation * 1.15, credit_lines, late_90, dependents)
    severe = create_borrower_data(age, income * 0.85, debt_ratio * 1.30, credit_utilisation * 1.25, credit_lines, late_90, dependents)

    normal_prob = model.predict_proba(normal)[0][1]
    mild_prob = model.predict_proba(mild)[0][1]
    moderate_prob = model.predict_proba(moderate)[0][1]
    severe_prob = model.predict_proba(severe)[0][1]

    normal_category = risk_category(normal_prob)
    severe_category = risk_category(severe_prob)
    rec = recommendation(normal_prob)

    increase = severe_prob - normal_prob

    st.subheader("Credit Risk Assessment Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="card">
        <div class="metric-title">Default Probability</div>
        <div class="metric-value">{normal_prob:.2%}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
        <div class="metric-title">Risk Category</div>
        <div class="metric-value {risk_class(normal_category)}">{normal_category}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="card">
        <div class="metric-title">Severe Stress Increase</div>
        <div class="metric-value">{increase:.2%}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="card">
        <div class="metric-title">Recommendation</div>
        <div class="metric-value">{rec}</div>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Borrower Input Summary")

    borrower_summary = pd.DataFrame({
        "Input Variable": [
            "Age",
            "Monthly Income",
            "Debt Ratio",
            "Credit Utilisation",
            "Open Credit Lines and Loans",
            "Times 90 Days Late",
            "Number of Dependents"
        ],
        "Value": [
            age,
            income,
            debt_ratio,
            credit_utilisation,
            credit_lines,
            late_90,
            dependents
        ]
    })

    st.dataframe(borrower_summary, use_container_width=True)

    st.subheader("Inflation Stress Testing Results")

    results = pd.DataFrame({
        "Scenario": ["Normal", "Mild Inflation", "Moderate Inflation", "Severe Inflation"],
        "Income Change": ["None", "-5%", "-10%", "-15%"],
        "Debt Ratio Change": ["None", "+10%", "+20%", "+30%"],
        "Credit Utilisation Change": ["None", "+5%", "+15%", "+25%"],
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

    st.subheader("Default Probability Across Inflation Scenarios")
    st.bar_chart(chart_data.set_index("Scenario"))

    st.subheader("Risk Migration Analysis")

    migration_data = pd.DataFrame({
        "Stage": ["Normal Condition", "Severe Inflation Condition"],
        "Risk Category": [normal_category, severe_category],
        "Default Probability": [f"{normal_prob:.2%}", f"{severe_prob:.2%}"]
    })

    st.dataframe(migration_data, use_container_width=True)

    if severe_category != normal_category:
        st.warning(
            f"The borrower migrates from {normal_category} under normal conditions to {severe_category} under severe inflation. "
            "This suggests that inflation stress testing can reveal additional lending risk not visible under normal assessment."
        )
    else:
        st.info(
            f"The borrower remains in the {normal_category} category under severe inflation, although the predicted probability changes."
        )

    st.subheader("Interpretation")

    if increase > 0:
        st.warning(
            f"Under severe inflation, the predicted default probability increases by {increase:.2%}. "
            "This indicates that inflationary pressure may weaken borrower repayment capacity and should be considered before final lending approval."
        )
    else:
        st.success(
            "The model does not show an increase in default probability under severe inflation for this borrower profile."
        )

    with st.expander("Model Information"):
        st.write("Model used: Gradient Boosting Classifier")
        st.write("ROC-AUC: 0.867")
        st.write("Top predictors identified during model development:")
        st.write("- Number of Times 90 Days Late")
        st.write("- Revolving Utilisation of Unsecured Lines")
        st.write("- Number of Times 60–89 Days Past Due")
        st.write("- Number of Times 30–59 Days Past Due")
        st.write("- Age")
        st.write("- Debt Ratio")

    with st.expander("Prototype Disclaimer"):
        st.write(
            "This application is a dissertation prototype designed to demonstrate an inflation-aware AI credit risk decision support system. "
            "It should not be used for real lending decisions without regulatory validation, explainability testing, fairness assessment, secure authentication and private deployment."
        )
