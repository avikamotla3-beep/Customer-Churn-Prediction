import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load trained model
MODEL_PATH = "models/telecom_churn_model.pkl"
THRESHOLD_PATH = "models/churn_threshold.pkl"

model = joblib.load(MODEL_PATH)
threshold = joblib.load(THRESHOLD_PATH)

# Application title
st.title("📊 Telecom Customer Churn Prediction")
st.write(
    "An AI-powered application to predict customer churn "
    "and support customer retention."
)

st.divider()

# Customer information
st.header("Enter Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (Months)", 0, 72, 12)

with col2:
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox(
        "Multiple Lines", ["Yes", "No", "No phone service"]
    )
    internet = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )
    security = st.selectbox(
        "Online Security", ["Yes", "No", "No internet service"]
    )
    backup = st.selectbox(
        "Online Backup", ["Yes", "No", "No internet service"]
    )

with col3:
    protection = st.selectbox(
        "Device Protection", ["Yes", "No", "No internet service"]
    )
    support = st.selectbox(
        "Tech Support", ["Yes", "No", "No internet service"]
    )
    tv = st.selectbox(
        "Streaming TV", ["Yes", "No", "No internet service"]
    )
    movies = st.selectbox(
        "Streaming Movies", ["Yes", "No", "No internet service"]
    )

st.divider()
st.header("Billing and Contract Information")

col4, col5 = st.columns(2)

with col4:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

with col5:
    monthly = st.number_input(
        "Monthly Charges", min_value=0.0, value=70.0
    )
    total = st.number_input(
        "Total Charges", min_value=0.0, value=840.0
    )

st.divider()

# Prediction
if st.button("Predict Customer Churn", type="primary"):

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    probability = model.predict_proba(customer)[0][1]

    if probability >= threshold:
        st.error("⚠️ Customer is likely to churn!")
        st.write("Recommended action: Consider a customer-retention offer.")
    else:
        st.success("✅ Customer is likely to stay!")

    st.metric("Churn Probability", f"{probability * 100:.2f}%")

    st.progress(float(probability))

    st.caption(f"Classification threshold: {threshold}")