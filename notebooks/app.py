import streamlit as st
import joblib

# Load trained model and feature columns
model = joblib.load("rf_churn_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn.")

st.success("Model loaded successfully!")

st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )

with col2:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=50.0
    )

with col3:
    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=500.0
    )