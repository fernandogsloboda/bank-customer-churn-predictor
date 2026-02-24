# 🛡️ Bank Customer Churn Intelligence

### 🔴 **Live Demo:** [Link to your Streamlit App]

## 🎯 Strategic Overview
Customer attrition (Churn) is one of the most significant financial drains in the banking sector. This project delivers a high-performance **Predictive Analytics System** designed to identify at-risk customers with **82% accuracy** and a **69% recall rate** for churners.

Unlike standard models, this system focuses on **Business Impact**, providing strategic recommendations based on customer behavior, age brackets, and financial standing.

---

## 🚀 Technical Highlights

### 1. Advanced Machine Learning Pipeline
* **Algorithm:** Utilized **XGBoost** (Extreme Gradient Boosting) for state-of-the-art classification performance.
* **Data Imbalance Handling:** Implemented **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the dataset, increasing the model's ability to detect actual churners by over 40% compared to baseline models.
* **Feature Engineering:** Processed categorical variables via One-Hot Encoding and Label Encoding for optimal model ingestion.

### 2. Interactive Decision Support System (Streamlit)
* **Real-time Prediction:** Instant churn probability scoring based on user input.
* **Risk Drivers Visualization:** Dynamic bar charts showing the top variables influencing each specific prediction.
* **Actionable Insights:** Automated strategic suggestions (e.g., retention campaigns for inactive members or high-net-worth individuals).

---

## 🛠️ Tech Stack
* **Language:** Python 3.13
* **Modeling:** XGBoost, Scikit-Learn
* **Data Handling:** Pandas, NumPy, Imbalanced-Learn (SMOTE)
* **Visualization:** Matplotlib, Seaborn
* **Deployment:** Streamlit Cloud

---

## ⚙️ Quick Start (Local Execution)

```bash
git clone [https://github.com/fernandogsloboda/bank-customer-churn-predictor.git](https://github.com/fernandogsloboda/bank-customer-churn-predictor.git) && cd bank-customer-churn-predictor && pip install -r requirements.txt && streamlit run churn_app.py