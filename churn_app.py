import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# --- Page Setup ---
st.set_page_config(page_title="Alpha Bank | Churn Intel", layout="wide")

@st.cache_resource
def load_assets():
    # Loading model, features list, and label encoder
    pack = joblib.load('churn_model_pack.pkl')
    return pack['model'], pack['features']

model, features = load_assets()

# --- Custom Styling ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stStatus { font-size: 1.2rem; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.title("🏛️ Alpha Bank Intelligence")
st.subheader("Customer Attrition Predictive System")

# --- Input Section ---
col_in, col_out = st.columns([1, 1.5], gap="large")

with col_in:
    st.write("### 👤 Customer Profile")
    with st.form("churn_form"):
        c1, c2 = st.columns(2)
        with c1:
            credit = st.number_input("Credit Score", 300, 850, 650)
            age = st.number_input("Age", 18, 90, 40)
            tenure = st.slider("Tenure", 0, 10, 5)
            balance = st.number_input("Balance ($)", 0.0, 250000.0, 75000.0)
        with c2:
            prods = st.selectbox("Products", [1, 2, 3, 4])
            card = st.selectbox("Has Card?", ["Yes", "No"])
            active = st.selectbox("Active?", ["Yes", "No"])
            salary = st.number_input("Salary ($)", 0.0, 200000.0, 100000.0)
        
        geo = st.selectbox("Country", ["France", "Germany", "Spain"])
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        
        run = st.form_submit_button("Analyze Risk Profile")

# --- Prediction & Visualization ---
if run:
    # Feature engineering for prediction
    input_data = pd.DataFrame([{
        'CreditScore': credit, 'Gender': 1 if gender == "Male" else 0,
        'Age': age, 'Tenure': tenure, 'Balance': balance,
        'NumOfProducts': prods, 'HasCrCard': 1 if card == "Yes" else 0,
        'IsActiveMember': 1 if active == "Yes" else 0,
        'EstimatedSalary': salary, 'Geography_Germany': 1 if geo == "Germany" else 0,
        'Geography_Spain': 1 if geo == "Spain" else 0
    }])
    input_data = input_data[features]
    
    prob = model.predict_proba(input_data)[0][1]

    with col_out:
        st.write("### 📊 Diagnostic Dashboard")
        
        # Risk Gauge Simulation
        st.progress(prob)
        if prob > 0.5:
            st.error(f"CRITICAL RISK: {prob:.1%} probability of exit.")
        else:
            st.success(f"STABLE PROFILE: {prob:.1%} probability of exit.")

        tab1, tab2 = st.tabs(["Risk Drivers", "Contextual Analysis"])

        with tab1:
            # Local feature importance
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=features).sort_values()
            
            fig, ax = plt.subplots(figsize=(8, 4))
            feat_imp.tail(5).plot(kind='barh', color='#1f77b4', ax=ax)
            ax.set_title("Top 5 Variables Impacting this Prediction")
            st.pyplot(fig)

        with tab2:
            # Benchmarking visualization
            st.write("#### Balance Comparison")
            avg_balance = 76485.89 # Data from our EDA
            fig2, ax2 = plt.subplots(figsize=(8, 2))
            ax2.barh(["Bank Avg", "Current Client"], [avg_balance, balance], color=['#bdc3c7', '#2980b9'])
            st.pyplot(fig2)
            
            if balance > avg_balance:
                st.info(f"This client holds {(balance - avg_balance):,.2f} above the bank average.")

        # Business Insights
        st.write("### 💡 Strategic Actions")
        if prob > 0.5:
            if age > 45: st.warning("- Priority: Invite for private wealth management session.")
            if prods == 1: st.warning("- Retention: Cross-sell life insurance or secondary card.")
        else:
            st.info("- Action: Pre-approve for higher credit limit or premium tier.")
