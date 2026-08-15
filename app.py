import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL, SCALER AND COLUMNS
# --------------------------------------------------

model = joblib.load("telco_churn_model_v2.pkl")
scaler = joblib.load("scaler_v2.pkl")
columns = joblib.load("columns_v2.pkl")


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title(" Telco Customer Churn Prediction")

st.write(
    "Enter the customer's information below to predict "
    "whether the customer is likely to churn."
)


# --------------------------------------------------
# CUSTOMER INFORMATION
# --------------------------------------------------

st.header(" Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

with col2:
    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

with col3:
    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )


col1, col2, col3 = st.columns(3)

with col1:
    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

with col2:
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )

with col3:
    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )


# --------------------------------------------------
# SERVICES
# --------------------------------------------------

st.header(" Services")

col1, col2, col3 = st.columns(3)

with col1:
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

with col2:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

with col3:
    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )


col1, col2, col3 = st.columns(3)

with col1:
    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

with col2:
    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

with col3:
    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )


col1, col2 = st.columns(2)

with col1:
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

with col2:
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )


# --------------------------------------------------
# CONTRACT AND BILLING
# --------------------------------------------------

st.header(" Contract & Billing")

col1, col2, col3 = st.columns(3)

with col1:
    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

with col2:
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

with col3:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


col1, col2 = st.columns(2)

with col1:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=1.0
    )

with col2:
    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1000.0,
        step=10.0
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

st.divider()

if st.button(
    " Predict Churn",
    use_container_width=True
):

    # Create dataframe
    customer = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })


    # Same encoding used during training
    customer_encoded = pd.get_dummies(
        customer,
        drop_first=True
    )


    # Make columns exactly match training columns
    customer_encoded = customer_encoded.reindex(
        columns=columns,
        fill_value=0
    )


    # Apply saved scaler
    customer_scaled = scaler.transform(
        customer_encoded
    )


    # Prediction
    prediction = model.predict(
        customer_scaled
    )[0]


    # Probability
    probability = model.predict_proba(
        customer_scaled
    )[0][1]


    probability_percent = probability * 100


    # --------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------

    st.header(" Prediction Result")


    if prediction == 1:

        st.error(
            " Customer is likely to CHURN"
        )

    else:

        st.success(
            " Customer is likely to STAY"
        )


    st.metric(
        "Churn Probability",
        f"{probability_percent:.2f}%"
    )


    st.progress(
        int(probability_percent)
    )


    if probability >= 0.5:

        st.warning(
            "This customer has a relatively high "
            "risk of churn."
        )

    else:

        st.info(
            "This customer has a relatively low "
            "risk of churn."
        )