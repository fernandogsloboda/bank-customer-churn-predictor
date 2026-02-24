# 🏛️ Alpha Bank | Customer Churn Intelligence

### 🔴 **Live Application:** [[LINK_DO_APP](https://bank-customer-churn-predictor-fpdytvyqt8n9lmdzbzqgqs.streamlit.app/#12-0)]

## 🎯 Strategic Overview
Customer attrition (Churn) is a critical financial drain in the banking sector. This project delivers an **End-to-End Machine Learning Solution** to predict the probability of a customer leaving the bank. 

By leveraging high-performance algorithms, the system achieved an **82% overall accuracy** with a focused **69% Recall on churners**, identifying nearly 70% of at-risk clients before they depart.

---

## 🚀 Technical Deep Dive

### 1. The Engine: XGBoost (Extreme Gradient Boosting)
Instead of a single decision model, I implemented **XGBoost**. It works through an iterative process called *Boosting*, where a sequence of hundreds of decision trees is built. Each new tree is specifically designed to correct the residual errors made by the previous ones, resulting in a highly precise and robust risk score.



### 2. Solving Data Imbalance: SMOTE
A common challenge in Churn datasets is the lack of "churn" examples (only 20% of the base). To prevent the model from becoming biased towards active customers, I applied **SMOTE** (Synthetic Minority Over-sampling Technique). This technique creates synthetic data points for the minority class, teaching the model to recognize subtle churn patterns that would otherwise be ignored.



### 3. Intelligent Dashboard & Interpretability
* **Real-time Scoring:** Instant churn probability calculation.
* **Feature Importance:** Visual bar charts explaining the "why" behind each prediction, highlighting variables like Age, Balance, and Product count.
* **Business Logic:** Automated strategic recommendations based on the predicted risk level and customer profile.

---

## 🛠️ Tech Stack
* **Language:** Python 3.13
* **Machine Learning:** XGBoost, Scikit-Learn
* **Data Prep:** Pandas, NumPy, Imbalanced-Learn (SMOTE)
* **Framework:** Streamlit (Web App)
* **Visualization:** Matplotlib, Seaborn

---

## ⚙️ How to Run Locally

```bash
git clone [https://github.com/fernandogsloboda/bank-customer-churn-predictor.git](https://github.com/fernandogsloboda/bank-customer-churn-predictor.git)
cd bank-customer-churn-predictor
pip install -r requirements.txt
streamlit run churn_app.py
