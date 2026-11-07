import streamlit as st
import pandas as pd
import joblib
import plotly.express as px


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ==================================================
# CUSTOM STYLING
# ==================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fc;
    }

    h1 {
        color: #17365d;
    }

    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# LOAD MODEL AND DATASET
# ==================================================

@st.cache_resource
def load_model():
    model = joblib.load("models/telecom_churn_model.pkl")
    threshold = joblib.load("models/churn_threshold.pkl")
    return model, threshold


@st.cache_data
def load_data():
    data = pd.read_csv(
        "dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce"
    )

    return data


model, threshold = load_model()
df = load_data()


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("📊 Telecom Churn Analytics")

st.sidebar.write(
    "Customer Churn Prediction System"
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview Dashboard",
        "Predict Customer Churn",
        "Model Performance"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Machine Learning Project\n\n"
    "Built using Python, Scikit-learn and Streamlit."
)


# ==================================================
# PAGE 1: OVERVIEW DASHBOARD
# ==================================================

if page == "Overview Dashboard":

    st.title("📊 Telecom Customer Churn Dashboard")

    st.write(
        "Explore customer data, churn patterns and "
        "important business insights."
    )

    total_customers = len(df)

    churned_customers = (
        df["Churn"].astype(str).str.strip() == "Yes"
    ).sum()

    retained_customers = total_customers - churned_customers

    churn_rate = (
        churned_customers / total_customers
    ) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Churned Customers",
        f"{churned_customers:,}"
    )

    col3.metric(
        "Retained Customers",
        f"{retained_customers:,}"
    )

    col4.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )

    st.divider()

    # Churn Distribution

    st.subheader("Customer Churn Distribution")

    churn_counts = (
        df["Churn"]
        .value_counts()
        .rename_axis("Customer Status")
        .reset_index(name="Number of Customers")
    )

    fig1 = px.pie(
        churn_counts,
        names="Customer Status",
        values="Number of Customers",
        title="Customers Who Stayed vs Churned",
        hole=0.45
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # Churn by Contract

    st.subheader("Churn by Contract Type")

    contract_data = (
        df.groupby(["Contract", "Churn"])
        .size()
        .reset_index(name="Customers")
    )

    fig2 = px.bar(
        contract_data,
        x="Contract",
        y="Customers",
        color="Churn",
        barmode="group",
        title="Customer Churn Across Contract Types"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # Churn by Internet Service

    st.subheader("Churn by Internet Service")

    internet_data = (
        df.groupby(["InternetService", "Churn"])
        .size()
        .reset_index(name="Customers")
    )

    fig3 = px.bar(
        internet_data,
        x="InternetService",
        y="Customers",
        color="Churn",
        barmode="group",
        title="Churn Across Internet Services"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    # Monthly Charges

    st.subheader("Monthly Charges Distribution")

    fig4 = px.histogram(
        df,
        x="MonthlyCharges",
        color="Churn",
        nbins=30,
        barmode="overlay",
        title="Monthly Charges and Customer Churn"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )


# ==================================================
# PAGE 2: CUSTOMER CHURN PREDICTION
# ==================================================

elif page == "Predict Customer Churn":

    st.title("🎯 Customer Churn Prediction")

    st.caption(
        "Enter customer details to estimate their churn probability."
    )

    st.subheader("Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12
        )

    with col2:

        phone = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

    with col3:

        protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

    st.divider()

    st.subheader("Billing and Contract Information")

    col4, col5 = st.columns(2)

    with col4:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

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
            "Monthly Charges",
            min_value=0.0,
            value=70.0
        )

        total = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0
        )

    st.divider()

    # Prediction Button

    if st.button(
        "Predict Customer Churn",
        type="primary",
        use_container_width=True
    ):

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

        prediction = int(probability >= threshold)

        st.divider()

        st.subheader("Prediction Results")

        if probability >= 0.60:

            risk = "High"

            st.error("🔴 HIGH CHURN RISK")

            recommendation = (
                "Prioritise this customer for a retention campaign. "
                "Consider a personalised discount, contract upgrade "
                "or proactive customer-support call."
            )

        elif probability >= 0.35:

            risk = "Medium"

            st.warning("🟡 MEDIUM CHURN RISK")

            recommendation = (
                "Monitor customer satisfaction, review service usage "
                "and offer relevant support before renewal."
            )

        else:

            risk = "Low"

            st.success("🟢 LOW CHURN RISK")

            recommendation = (
                "Maintain service quality and customer engagement. "
                "Consider loyalty benefits to encourage retention."
            )

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

        r2.metric(
            "Customer Risk Category",
            risk
        )

        r3.metric(
            "Model Prediction",
            "Likely to Churn" if prediction else "Likely to Stay"
        )

        st.progress(float(probability))

        st.subheader("💡 Recommended Business Action")

        st.info(recommendation)

        st.caption(
            f"Model classification threshold: {threshold:.2f}. "
            "Risk bands are illustrative and can be adjusted "
            "according to business requirements."
        )


# ==================================================
# PAGE 3: MODEL PERFORMANCE
# ==================================================

elif page == "Model Performance":

    st.title("📈 Machine Learning Model Performance")

    st.caption(
        "Evaluation results from the final held-out test dataset."
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Accuracy", "76.30%")
    m2.metric("Precision", "54.08%")
    m3.metric("Recall", "70.86%")
    m4.metric("ROC-AUC", "84.06%")

    st.divider()

    st.subheader("Final Classification Report")

    report = pd.DataFrame({
        "Metric": [
            "Precision",
            "Recall",
            "F1-Score"
        ],
        "Stayed": [
            0.88,
            0.78,
            0.83
        ],
        "Churned": [
            0.54,
            0.71,
            0.61
        ]
    })

    st.dataframe(
        report.style.format({
            "Stayed": "{:.2f}",
            "Churned": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Confusion Matrix")

    confusion = pd.DataFrame(
        [[810, 225], [109, 265]],
        index=[
            "Actual Stayed",
            "Actual Churned"
        ],
        columns=[
            "Predicted Stayed",
            "Predicted Churned"
        ]
    )

    fig = px.imshow(
        confusion,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Final Model Confusion Matrix"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        "Final Model: Tuned Logistic Regression | "
        "Decision Threshold: 0.60 | "
        "Test Samples: 1,409"
    )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Telecom Customer Churn Prediction | "
    "Machine Learning Project | Built with Python and Streamlit"
)