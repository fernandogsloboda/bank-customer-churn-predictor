import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Carregar o cérebro da IA
@st.cache_resource
def load_model():
    return joblib.load('churn_model_pack.pkl')

data_pack = load_model()
model = data_pack['model']
features = data_pack['features']

st.set_page_config(page_title="Churn Predictor Pro", layout="centered")

st.title("🛡️ Bank Churn Prediction System")
st.markdown("Enter customer details to predict the probability of departure (Churn).")

# 2. Formulário de Entrada
with st.form("prediction_form"):
    st.subheader("👤 Customer Profile")
    col1, col2 = st.columns(2)
    
    with col1:
        credit_score = st.number_input("Credit Score", 300, 850, 600)
        age = st.number_input("Age", 18, 100, 35)
        tenure = st.slider("Tenure (Years as customer)", 0, 10, 5)
        balance = st.number_input("Account Balance ($)", 0.0, 500000.0, 50000.0)
        
    with col2:
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_card = st.radio("Has Credit Card?", ["Yes", "No"])
        is_active = st.radio("Is Active Member?", ["Yes", "No"])
        salary = st.number_input("Estimated Annual Salary ($)", 0.0, 500000.0, 100000.0)
        gender = st.selectbox("Gender", ["Male", "Female"])
        geo = st.selectbox("Country", ["France", "Germany", "Spain"])

    submit = st.form_submit_button("Predict Risk")

# 3. Lógica de Previsão
if submit:
    # Preparar os dados exatamente como o modelo treinou
    input_data = pd.DataFrame([{
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
    
    # Garantir a ordem das colunas
    input_data = input_data[features]
    
    # Predição
    prob = model.predict_proba(input_data)[0][1]
    
    st.markdown("---")
    if prob > 0.5:
        st.error(f"⚠️ **HIGH RISK:** This customer has a {prob:.1%} probability of leaving.")
        st.write("👉 **Recommendation:** Offer a loyalty bonus or personal account manager contact.")
    else:
        st.success(f"✅ **LOW RISK:** This customer has only a {prob:.1%} probability of leaving.")
        st.write("👉 **Recommendation:** Maintain standard relationship and offer cross-sell products.")