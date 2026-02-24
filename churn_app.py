import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# --- Page Setup ---
st.set_page_config(page_title="Alpha Bank | Churn Intel", layout="wide")

@st.cache_resource
def load_assets():
    try:
        pack = joblib.load('churn_model_pack.pkl')
        return pack['model'], pack['features']
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        return None, None

model, features = load_assets()

# --- Helper Functions ---
def get_metric_style(prob):
    if prob < 0.3: return "#28a745" # Green
    elif prob < 0.6: return "#ffc107" # Yellow
    return "#dc3545" # Red

# --- Styling ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
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

if run and model is not None:
    # Prep data for XGBoost
    input_df = pd.DataFrame([{
        'CreditScore': credit, 'Gender': 1 if gender == "Male" else 0,
        'Age': age, 'Tenure': tenure, 'Balance': balance,
        'NumOfProducts': prods, 'HasCrCard': 1 if card == "Yes" else 0,
        'IsActiveMember': 1 if active == "Yes" else 0,
        'EstimatedSalary': salary, 'Geography_Germany': 1 if geo == "Germany" else 0,
        'Geography_Spain': 1 if geo == "Spain" else 0
    }])
    input_df = input_df[features]
    
    # Predict probability
    prob = float(model.predict_proba(input_df)[0][1])
    risk_color = get_metric_style(prob)

    # --- Metrics Layout (Centered) ---
    m_empty1, m1, m2, m_empty2 = st.columns([1, 2, 2, 1])
    
    with m1:
        st.markdown(f"<h5 style='text-align: center; color: grey;'>Risk Score</h5>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center; color: {risk_color};'>{prob:.1%}</h2>", unsafe_allow_html=True)
    with m2:
        status = "CRITICAL" if prob > 0.6 else ("WARNING" if prob > 0.3 else "STABLE")
        st.markdown(f"<h5 style='text-align: center; color: grey;'>Profile Status</h5>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center; color: {risk_color};'>{status}</h2>", unsafe_allow_html=True)

    st.markdown("---")

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.write("### 📉 Risk Analysis")
        st.progress(prob)
        
        if prob > 0.6: st.error(f"⚠️ HIGH CHURN RISK: {prob:.1%}")
        elif prob > 0.3: st.warning(f"⚡ MODERATE RISK: {prob:.1%}")
        else: st.success(f"✅ LOW RISK: {prob:.1%}")

        st.write("### 💡 Strategic Recommendations")
        if prob > 0.3:
            if age > 45: st.warning("- Priority: Personal financial advisor assignment.")
            if active == "No": st.warning("- Campaign: Direct reactivation offer.")
            if prods == 1: st.warning("- Retention: Product bundle discount.")
        else:
            st.info("- Action: Standard maintenance. Offer premium upgrades.")

    with col_r:
        tab1, tab2 = st.tabs(["Decision Drivers", "Financial Benchmark"])
        with tab1:
            # Importance visualization
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=features).sort_values()
            fig, ax = plt.subplots(figsize=(8, 5))
            feat_imp.tail(5).plot(kind='barh', color=risk_color, ax=ax)
            ax.set_title("Top Variables Influencing Prediction")
            st.pyplot(fig)
        with tab2:
            # Compare with average balance
            avg_balance = 76485.89 
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            ax2.barh(["Bank Avg", "Current Client"], [avg_balance, balance], color=['#95a5a6', risk_color])
            st.pyplot(fig2)
            st.write(f"Difference: **${(balance - avg_balance):+,.2f}**")
else:
    st.info("👈 Enter customer details in the sidebar and click 'Analyze Risk' to begin.")
