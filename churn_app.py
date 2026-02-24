import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# --- Page Setup ---
st.set_page_config(page_title="Alpha Bank | Churn Intel", layout="wide")

@st.cache_resource
def load_assets():
    # Load model and feature list
    pack = joblib.load('churn_model_pack.pkl')
    return pack['model'], pack['features']

model, features = load_assets()

# --- Custom Styling & Dynamic Colors ---
def get_metric_style(prob):
    if prob < 0.3:
        return "#28a745" # Green (Good)
    elif prob < 0.6:
        return "#ffc107" # Yellow (Medium)
    else:
        return "#dc3545" # Red (Bad)

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    /* Style for metric boxes to ensure text visibility */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
    }
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

if run:
    # Data transformation
    input_df = pd.DataFrame([{
        'CreditScore': credit, 'Gender': 1 if gender == "Male" else 0,
        'Age': age, 'Tenure': tenure, 'Balance': balance,
        'NumOfProducts': prods, 'HasCrCard': 1 if card == "Yes" else 0,
        'IsActiveMember': 1 if active == "Yes" else 0,
        'EstimatedSalary': salary, 'Geography_Germany': 1 if geo == "Germany" else 0,
        'Geography_Spain': 1 if geo == "Spain" else 0
    }])
    input_df = input_df[features]
    
    # Prediction
    prob = model.predict_proba(input_df)[0][1]
    risk_color = get_metric_style(prob)

    # --- Header Metrics with Forced Colors ---
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.markdown(f"<h5 style='color: grey;'>Risk Score</h5>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='color: {risk_color};'>{prob:.1%}</h2>", unsafe_allow_html=True)
    with m2:
        status = "CRITICAL" if prob > 0.6 else ("WARNING" if prob > 0.3 else "STABLE")
        st.markdown(f"<h5 style='color: grey;'>Profile Status</h5>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='color: {risk_color};'>{status}</h2>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<h5 style='color: grey;'>Potential Loss</h5>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='color: #2c3e50;'>${balance:,.2f}</h2>", unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.write("### 📉 Risk Analysis")
        st.progress(float(prob))
        
        if prob > 0.6:
            st.error(f"⚠️ HIGH CHURN RISK: Probability {prob:.1%}")
        elif prob > 0.3:
            st.warning(f"⚡ MODERATE RISK: Probability {prob:.1%}")
        else:
            st.success(f"✅ LOW RISK: Probability {prob:.1%}")

        st.write("### 💡 Strategic Recommendations")
        if prob > 0.3:
            if age > 45: st.warning("- Schedule priority financial planning call.")
            if prods == 1: st.warning("- Offer multi-product bundle discounts.")
            if active == "No": st.warning("- Send personalized engagement email.")
        else:
            st.info("- Customer is stable. Eligible for premium credit line increases.")

    with col_right:
        tab1, tab2 = st.tabs(["Decision Drivers", "Financial Benchmark"])
        
        with tab1:
            # Model explainability
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=features).sort_values()
            fig, ax = plt.subplots(figsize=(8, 5))
            feat_imp.tail(5).plot(kind='barh', color=risk_color, ax=ax)
            ax.set_title("Top Variables Influencing Prediction")
            st.pyplot(fig)

        with tab2:
            # Comparison against average
            avg_balance = 76485.89 
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            ax2.barh(["Bank Avg", "Current Client"], [avg_balance, balance], color=['#95a5a6', risk_color])
            ax2.set_title("Balance Comparison ($)")
            st.pyplot(fig2)
            
            diff = balance - avg_balance
            st.write(f"Difference from Average: **${diff:+,.2f}**")
else:
    st.info("👈 Enter customer details in the sidebar and click 'Analyze Risk' to begin.")
