import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(page_title="Churn Intel Pro", layout="wide")

@st.cache_resource
def load_assets():
    pack = joblib.load('churn_model_pack.pkl')
    return pack['model'], pack['features']

model, features = load_assets()

# --- Header ---
st.title("🏛️ Alpha Bank | Customer Churn Intelligence")
st.markdown("Predictive analytics system to identify high-risk customer profiles.")

# --- Layout ---
col_input, col_display = st.columns([1, 2])

with col_input:
    st.subheader("Client Parameters")
    with st.form("input_form"):
        credit_score = st.number_input("Credit Score", 300, 850, 600)
        age = st.number_input("Age", 18, 90, 40)
        tenure = st.slider("Tenure (Years)", 0, 10, 5)
        balance = st.number_input("Balance ($)", 0.0, 250000.0, 50000.0)
        num_products = st.selectbox("Products", [1, 2, 3, 4])
        has_card = st.selectbox("Credit Card", ["Yes", "No"])
        is_active = st.selectbox("Active Member", ["Yes", "No"])
        salary = st.number_input("Salary ($)", 0.0, 200000.0, 100000.0)
        gender = st.selectbox("Gender", ["Male", "Female"])
        geo = st.selectbox("Geography", ["France", "Germany", "Spain"])
        
        submitted = st.form_submit_button("Run Analysis")

if submitted:
    # --- Data Prep ---
    input_df = pd.DataFrame([{
        'CreditScore': credit_score,
        'Gender': 1 if gender == "Male" else 0,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': 1 if has_card == "Yes" else 0,
        'IsActiveMember': 1 if is_active == "Yes" else 0,
        'EstimatedSalary': salary,
        'Geography_Germany': 1 if geo == "Germany" else 0,
        'Geography_Spain': 1 if geo == "Spain" else 0
    }])
    input_df = input_df[features]

    # --- Prediction ---
    prob = model.predict_proba(input_df)[0][1]

    with col_display:
        st.subheader("Diagnostic Result")
        
        # Risk Metric
        metric_color = "normal" if prob < 0.5 else "inverse"
        st.status(f"Churn Probability: {prob:.1%}", state="error" if prob > 0.5 else "complete")
        
        # Risk Drivers
        st.subheader("Key Risk Drivers")
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=features).sort_values()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        feat_imp.tail(5).plot(kind='barh', color='#2c3e50', ax=ax)
        st.pyplot(fig)

        # Strategic Insights
        st.subheader("Strategic Recommendations")
        if prob > 0.5:
            st.warning("⚠️ High departure risk detected. Engagement team should initiate retention protocol.")
            if age > 45: st.info("Targeted retirement or wealth management products recommended.")
            if is_active == "No": st.info("Account inactivity identified. Promotional reactivation campaign suggested.")
        else:
            st.success("✅ Stable profile. Eligible for cross-selling and premium services.")