# 🏛️ Alpha Bank | Customer Churn Intelligence

### 🔴 **Live Application:** [PASTE_YOUR_STREAMLIT_LINK_HERE]

## 🎯 Project Scope
Customer retention is a critical KPI for financial institutions. This project delivers an **End-to-End Machine Learning Solution** to predict the probability of a customer leaving the bank (Churn). 

By leveraging high-performance algorithms, we achieved an **82% overall accuracy** with a focused **69% Recall on churners**, ensuring the bank identifies nearly 70% of at-risk clients before they depart.

---

## 🚀 Key Technical Features

### 1. Data Science Pipeline
* **Engine:** Built with **XGBoost Classifier**, the industry standard for structured data.
* **Class Imbalance:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to the training set to prevent model bias towards the majority class.
* **Feature Engineering:** Automated processing of demographic and financial variables, including One-Hot Encoding for geographical data.

### 2. Intelligent Dashboard
* **Risk Score:** Real-time probability calculation based on customer behavior.
* **Interpretability:** Visual representation of **Feature Importance** to explain "the why" behind every prediction.
* **Benchmarking:** Automated comparison between individual client balance and the bank's overall average ($76.5k).
* **Strategic Insights:** Conditional logic providing specific business recommendations (e.g., Wealth Management vs. Re-activation campaigns).

---

## 🛠️ Tech Stack
* **Language:** Python 3.13
* **Machine Learning:** XGBoost, Scikit-Learn
* **Data Prep:** Pandas, NumPy, Imbalanced-Learn
* **Framework:** Streamlit (Web App)
* **Visualization:** Matplotlib, Seaborn

---

## ⚙️ How to Run Locally

```bash
git clone [https://github.com/fernandogsloboda/bank-customer-churn-predictor.git](https://github.com/fernandogsloboda/bank-customer-churn-predictor.git) && cd bank-customer-churn-predictor && pip install -r requirements.txt && streamlit run churn_app.py
