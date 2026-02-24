import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# --- Page Setup ---
st.set_page_config(page_title="Alpha Bank | Churn Intel", layout="wide")

@st.cache_resource
def load_assets():
    # Load model and feature list from the exported pickle
    pack = joblib.load('churn_model_pack.pkl')
    return pack['model'], pack['features']

model, features = load_assets()

# --- Custom Styling ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0px 2px 10px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar Inputs ---
st.sidebar.title("🏛️ Alpha Bank")
st.sidebar.markdown("### Customer Data Entry")

with st.sidebar.form("churn_form"):
    credit = st.number_input("Credit Score", 300, 850, 650)
    age = st.number_input("Age", 18, 90, 40)
    tenure = st.slider("Tenure (Years)", 0, 10, 5)
    balance = st.number_input("Account Balance ($)", 0.0, 250000.0, 75000.0)
    prods = st.selectbox("Number of Products", [1, 2, 3, 4])
    card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    active = st.selectbox("Active Member?", ["Yes", "No"])
    salary = st.number_input("Estimated Salary ($)", 0.0, 200000.0, 100000.0)
    geo = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    
    run = st.form_submit_button("ANALYZE RISK")

# --- Main Dashboard ---
st.title("📊 Customer Churn Intelligence Dashboard")
st.markdown("Real-time predictive analytics for client retention.")

if run:
    # Data transformation for XGBoost
    input_df = pd.DataFrame([{
        'CreditScore': credit, 'Gender': 1 if gender == "Male" else 0,
        'Age': age, 'Tenure': tenure, 'Balance': balance,
        'NumOfProducts': prods, 'HasCrCard': 1 if card == "Yes" else 0,
        'IsActiveMember': 1 if active == "Yes" else 0,
        'EstimatedSalary': salary, 'Geography_Germany': 1 if geo == "Germany" else 0,
        'Geography_Spain': 1 if geo == "Spain" else 0
    }])
    input_df = input_df[features]
    
    # Calculate probability
    prob = model.predict_proba(input_df)[0][1]

    # Header Metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Risk Score", f"{prob:.1%}")
    with m2:
        status = "CRITICAL" if prob > 0.5 else "STABLE"
        st.metric("Profile Status", status)
    with m3:
        st.metric("Target Variable", "Exited (Churn)")

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.write("### 📉 Risk Analysis")
        # Fixed: Casting to float and proper indentation
        st.progress(float(prob))
        
        if prob > 0.5:
            st.error(f"⚠️ HIGH CHURN RISK: This customer shows a {prob:.1%} probability of leaving.")
        else:
            st.success(f"✅ RETENTION LIKELY: This customer shows a {prob:.1%} probability of staying.")

        # Business Insights logic
        st.write("### 💡 Strategic Recommendations")
        if prob > 0.5:
            if age > 45: st.warning("- Schedule priority financial planning call.")
            if prods == 1: st.warning("- Offer multi-product bundle discounts.")
            if is_active == "No": st.warning("- Send personalized engagement email.")
        else:
            st.info("- Customer is stable. Eligible for premium credit line increases.")

    with col_right:
        tab1, tab2 = st.tabs(["Decision Drivers", "Financial Benchmark"])
        
        with tab1:
            # Local importance for the specific prediction
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=features).sort_values()
            fig, ax = plt.subplots(figsize=(8, 5))
            feat_imp.tail(5).plot(kind='barh', color='#3498db', ax=ax)
            ax.set_title("Top 5 Variables Impacting this Prediction")
            st.pyplot(fig)

        with tab2:
            # Benchmark analysis against bank average
            avg_balance = 76485.89 
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            colors = ['#95a5a6', '#e74c3c' if prob > 0.5 else '#2ecc71']
            ax2.barh(["Bank Avg", "Current Client"], [avg_balance, balance], color=colors)
            ax2.set_title("Balance Comparison ($)")
            st.pyplot(fig2)
            
            diff = balance - avg_balance
            if diff > 0:
                st.write(f"Client is **${diff:,.2f} above** the average bank balance.")
else:
    st.info("👈 Enter customer details in the sidebar and click 'Analyze Risk' to begin.")
