import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load churn prediction model
model = joblib.load("rf_churn_model.pkl")

# Load feature columns
feature_columns = joblib.load("feature_columns.pkl")

feature_name_map = {
    "SeniorCitizen": "Senior Citizen",
    "tenure": "Customer Tenure",
    "MonthlyCharges": "Monthly Charges",
    "TotalCharges": "Total Charges",

    "gender_Male": "Male Customer",
    "Partner_Yes": "Has Partner",
    "Dependents_Yes": "Has Dependents",
    "PhoneService_Yes": "Phone Service",
    "MultipleLines_Yes": "Multiple Phone Lines",

    "InternetService_Fiber optic": "Fiber Optic Internet",
    "InternetService_No": "No Internet Service",

    "OnlineSecurity_Yes": "Online Security",
    "OnlineBackup_Yes": "Online Backup",
    "DeviceProtection_Yes": "Device Protection",
    "TechSupport_Yes": "Tech Support",

    "StreamingTV_Yes": "Streaming TV",
    "StreamingMovies_Yes": "Streaming Movies",

    "PaperlessBilling_Yes": "Paperless Billing",

    "Contract_One year": "One-Year Contract",
    "Contract_Two year": "Two-Year Contract",

    "PaymentMethod_Electronic check": "Electronic Check Payment",
    "PaymentMethod_Mailed check": "Mailed Check Payment",
    "PaymentMethod_Bank transfer": "Bank Transfer Payment",
    "PaymentMethod_Credit card": "Credit Card Payment"
}
# Load LTV prediction model
ltv_model = joblib.load("ltv_model.pkl")

# Create SHAP explainer for the churn model
explainer = shap.TreeExplainer(model)
 
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
    st.subheader("👥 Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

with col2:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

with col3:
    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

col1, col2 = st.columns(2)

with col1:
    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

with col2:
    phone_service = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )
    st.subheader("📡 Services")

col1, col2, col3 = st.columns(3)

with col1:
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes"]
    )

with col2:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

with col3:
    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes"]
    )

col1, col2, col3 = st.columns(3)

with col1:
    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes"]
    )
with col2:
    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes"]
    )

with col3:
    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes"]
    )

col1, col2 = st.columns(2)

with col1:
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes"]
    )

with col2:
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes"]
    )

st.subheader("💳 Contract & Payment")

col1, col2, col3 = st.columns(3)

with col1:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

with col2:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer",
            "Credit card"
        ]
    )

with col3:
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )
st.divider()

st.subheader("🔮 Churn Prediction")

if st.button("Predict Churn", type="primary"):

    # ==========================================
    # CREATE INPUT DATA
    # ==========================================

    input_data = {
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,

        "gender_Male": 1 if gender == "Male" else 0,
        "Partner_Yes": 1 if partner == "Yes" else 0,
        "Dependents_Yes": 1 if dependents == "Yes" else 0,

        "PhoneService_Yes": 1 if phone_service == "Yes" else 0,
        "MultipleLines_Yes": 1 if multiple_lines == "Yes" else 0,

        "InternetService_Fiber optic":
            1 if internet_service == "Fiber optic" else 0,

        "InternetService_No":
            1 if internet_service == "No" else 0,

        "OnlineSecurity_Yes":
            1 if online_security == "Yes" else 0,

        "OnlineBackup_Yes":
            1 if online_backup == "Yes" else 0,

        "DeviceProtection_Yes":
            1 if device_protection == "Yes" else 0,

        "TechSupport_Yes":
            1 if tech_support == "Yes" else 0,

        "StreamingTV_Yes":
            1 if streaming_tv == "Yes" else 0,

        "StreamingMovies_Yes":
            1 if streaming_movies == "Yes" else 0,

        "PaperlessBilling_Yes":
            1 if paperless_billing == "Yes" else 0,

        "Contract_One year":
            1 if contract == "One year" else 0,

        "Contract_Two year":
            1 if contract == "Two year" else 0,

        "PaymentMethod_Electronic check":
            1 if payment_method == "Electronic check" else 0,

        "PaymentMethod_Mailed check":
            1 if payment_method == "Mailed check" else 0,

        "PaymentMethod_Bank transfer":
            1 if payment_method == "Bank transfer" else 0,

        "PaymentMethod_Credit card":
            1 if payment_method == "Credit card" else 0,
    }


    # ==========================================
    # CREATE DATAFRAME
    # ==========================================

    input_df = pd.DataFrame([input_data])


    # ==========================================
    # ADD ANY MISSING FEATURES
    # ==========================================

    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0


    # Keep same feature order as training
    input_df = input_df[feature_columns]


    # ==========================================
    # CHURN PREDICTION
    # ==========================================

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    probability_percent = probability * 100


    # ==========================================
    # LTV PREDICTION
    # ==========================================

    # Create input for LTV model
    ltv_input = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    # Predict Estimated Lifetime Value
    estimated_ltv = ltv_model.predict(ltv_input)[0]

    # Prevent negative LTV values
    estimated_ltv = max(0, estimated_ltv)


# ==========================================
# CHURN RISK ANALYSIS
# ==================================

risk_level = None

st.divider()

st.subheader("📊 Churn Risk Analysis")

if probability_percent >= 70:

    risk_level = "🔴 High Churn Risk"

    risk_message = (
        "This customer has a high likelihood "
        "of leaving the service."
    )

elif probability_percent >= 40:

    risk_level = "🟡 Moderate Churn Risk"

    risk_message = (
        "This customer shows a moderate likelihood "
        "of leaving the service."
    )

else:

    risk_level = "🟢 Low Churn Risk"

    risk_message = (
        "This customer has a relatively low likelihood "
        "of leaving the service."
    )


# Display risk
st.write(f"### {risk_level}")


# Display probability
st.metric(
    "Churn Probability",
    f"{probability_percent:.1f}%"
)


# Display Estimated LTV
st.metric(
    "Estimated Customer Lifetime Value",
    f"₹{estimated_ltv:.2f}"
)


# Progress bar
st.progress(int(probability_percent))

# Risk explanation
st.write(risk_message)

 # ==========================================
# EXPLAINABLE AI
# ==========================================

st.divider()

st.subheader("🧠 Explainable AI — Why This Prediction?")

# Calculate SHAP values
shap_explanation = explainer(input_df)

# Extract explanation for churn class
customer_shap = shap_explanation[0, :, 1]

# Create waterfall plot
fig, ax = plt.subplots(figsize=(10, 6))

shap.plots.waterfall(
    customer_shap,
    max_display=10,
    show=False
)

# Display plot in Streamlit
st.pyplot(fig, clear_figure=True)


   # ==========================================
# KEY FACTORS
# ==========================================

st.subheader("📌 Key Factors Behind This Prediction")

# Get SHAP values
shap_values_customer = customer_shap.values

feature_names = input_df.columns


# Create DataFrame
shap_df = pd.DataFrame({
    "Feature": feature_names,
    "SHAP Value": shap_values_customer
})


# Convert technical names to readable names
shap_df["Feature"] = shap_df["Feature"].map(
    feature_name_map
).fillna(
    shap_df["Feature"]
)


# Calculate absolute impact
shap_df["Impact"] = shap_df[
        "SHAP Value"
    ].abs()


# Sort by impact
shap_df = shap_df.sort_values(
        by="Impact",
        ascending=False
    )


# Factors increasing churn
risk_factors = shap_df[
        shap_df["SHAP Value"] > 0
    ].head(5)


    # Factors reducing churn
protective_factors = shap_df[
        shap_df["SHAP Value"] < 0
    ].head(5)


    # Create two columns
col1, col2 = st.columns(2)


    # ==========================================
    # INCREASING RISK
    # ==========================================

with col1:

        st.markdown("### 🔴 Increasing Churn Risk")


        if len(risk_factors) > 0:

            for _, row in risk_factors.iterrows():

                st.write(
                    f"• **{row['Feature']}** "
                    f"(impact: +{row['SHAP Value']:.3f})"
                )

        else:

            st.success(
                "No major factors increasing churn risk."
            )


    # ==========================================
# REDUCING RISK
# ==========================================

with col2:

    st.markdown("### 🟢 Reducing Churn Risk")

    if len(protective_factors) > 0:

        for _, row in protective_factors.iterrows():

            st.write(
                f"• **{row['Feature']}** "
                f"(impact: {row['SHAP Value']:.3f})"
            )

    else:

        st.info(
            "No major factors reducing churn risk."
        )

        # =========================================
# RETENTION RECOMMENDATIONS
# =========================================

st.divider()

st.subheader("🎯 Automated Retention Recommendations")

recommendations = []

# High churn risk recommendations
if probability_percent >= 70:
    recommendations.append(
        "🚨 Priority Action: Contact this customer immediately with a personalized retention offer."
    )

# Contract recommendation
if contract == "Month-to-month":
    recommendations.append(
        "📅 Offer an incentive to switch to a One year or Two year contract."
    )

# Technical support recommendation
if tech_support == "No":
    recommendations.append(
        "🛠️ Offer free or discounted Technical Support to improve customer satisfaction."
    )

# Online security recommendation
if online_security == "No":
    recommendations.append(
        "🔒 Offer an Online Security package or promotional upgrade."
    )

# Payment recommendation
if payment_method == "Electronic check":
    recommendations.append(
        "💳 Encourage switching to automatic payment methods for a smoother experience."
    )

# Display recommendations
if len(recommendations) > 0:

    st.write("### Recommended Actions")

    for recommendation in recommendations:
        st.info(recommendation)

else:
    st.success(
        "✅ This customer currently shows no major retention concerns."
    ) 