import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #0b0e14;
    color: #f5f7fa;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.main-title {
    font-size: 42px;
    font-weight: 700;
    color: #f5f7fa;
    margin-bottom: 5px;
}

.subtitle {
    color: #9ca3af;
    font-size: 16px;
    line-height: 1.6;
}

.section-title {
    font-size: 21px;
    font-weight: 650;
    color: #f5f7fa;
    margin-top: 20px;
    margin-bottom: 18px;
}

.section-number {
    color: #36d6c5;
    font-size: 14px;
    font-weight: 700;
    margin-right: 8px;
}

.risk-card {
    background-color: #151922;
    border: 1px solid #292f3b;
    border-radius: 18px;
    padding: 28px;
    min-height: 310px;
}

.risk-title {
    color: #8d95a5;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.5px;
}

.risk-number {
    color: #f5f7fa;
    font-size: 48px;
    font-weight: 750;
    margin-top: 20px;
}

.risk-description {
    color: #9ca3af;
    line-height: 1.6;
}

.result-high {
    background-color: #2a1518;
    border: 1px solid #74333b;
    border-radius: 14px;
    padding: 18px;
    color: #ff9da6;
    font-weight: 650;
    margin-top: 18px;
}

.result-low {
    background-color: #102421;
    border: 1px solid #21645d;
    border-radius: 14px;
    padding: 18px;
    color: #63e6d6;
    font-weight: 650;
    margin-top: 18px;
}

.stButton > button {
    width: 100%;
    height: 54px;
    border-radius: 12px;
    border: none;
    background-color: #36d6c5;
    color: #07110f;
    font-size: 17px;
    font-weight: 700;
}

.stButton > button:hover {
    background-color: #4be1d0;
    color: #07110f;
}

div[data-baseweb="select"] > div {
    background-color: #151922;
    border-color: #292f3b;
    border-radius: 10px;
}

div[data-testid="stNumberInput"] input {
    background-color: #151922;
    color: #f5f7fa;
    border-radius: 10px;
}

label {
    color: #aeb5c2 !important;
}

hr {
    border-color: #242936;
}

.badge {
    display: inline-block;
    background-color: #153d39;
    color: #57dfcf;
    padding: 8px 15px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL, SCALER AND COLUMNS
# ============================================================

@st.cache_resource
def load_files():

    model = joblib.load("telco_churn_model_v2.pkl")

    scaler = joblib.load("scaler_v2.pkl")

    columns = joblib.load("columns_v2.pkl")

    return model, scaler, columns


model, scaler, columns = load_files()


# ============================================================
# HEADER
# ============================================================

col1, col2 = st.columns([4, 1])

with col1:

    st.markdown(
        '<div class="main-title">Telco Churn Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Customer retention intelligence tool. Enter customer information '
        'to estimate the probability of churn using a machine learning model.'
        '</div>',
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        '<div style="text-align:right; margin-top:15px;">'
        '<span class="badge">ML • Customer Retention</span>'
        '</div>',
        unsafe_allow_html=True
    )


st.divider()


# ============================================================
# MAIN PAGE
# ============================================================

left_column, right_column = st.columns([2.7, 1])


# ============================================================
# LEFT SIDE
# ============================================================

with left_column:

    # ========================================================
    # CUSTOMER INFORMATION
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '<span class="section-number">01</span>'
        'Customer Information'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        gender = st.selectbox(
            "Gender",
            [
                "Select...",
                "Female",
                "Male"
            ]
        )

    with col2:

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [
                "Select...",
                0,
                1
            ]
        )

    with col3:

        partner = st.selectbox(
            "Partner",
            [
                "Select...",
                "Yes",
                "No"
            ]
        )


    col1, col2, col3 = st.columns(3)

    with col1:

        dependents = st.selectbox(
            "Dependents",
            [
                "Select...",
                "Yes",
                "No"
            ]
        )

    with col2:

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12,
            step=1
        )

    with col3:

        phone_service = st.selectbox(
            "Phone Service",
            [
                "Select...",
                "Yes",
                "No"
            ]
        )


    # ========================================================
    # SERVICES
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '<span class="section-number">02</span>'
        'Services'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "Select...",
                "Yes",
                "No",
                "No phone service"
            ]
        )

    with col2:

        internet_service = st.selectbox(
            "Internet Service",
            [
                "Select...",
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

    with col3:

        online_security = st.selectbox(
            "Online Security",
            [
                "Select...",
                "Yes",
                "No",
                "No internet service"
            ]
        )


    col1, col2, col3 = st.columns(3)

    with col1:

        online_backup = st.selectbox(
            "Online Backup",
            [
                "Select...",
                "Yes",
                "No",
                "No internet service"
            ]
        )

    with col2:

        device_protection = st.selectbox(
            "Device Protection",
            [
                "Select...",
                "Yes",
                "No",
                "No internet service"
            ]
        )

    with col3:

        tech_support = st.selectbox(
            "Tech Support",
            [
                "Select...",
                "Yes",
                "No",
                "No internet service"
            ]
        )


    col1, col2, col3 = st.columns(3)

    with col1:

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "Select...",
                "Yes",
                "No",
                "No internet service"
            ]
        )

    with col2:

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "Select...",
                "Yes",
                "No",
                "No internet service"
            ]
        )

    with col3:

        st.empty()


    # ========================================================
    # CONTRACT AND BILLING
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '<span class="section-number">03</span>'
        'Contract & Billing'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        contract = st.selectbox(
            "Contract",
            [
                "Select...",
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

    with col2:

        paperless_billing = st.selectbox(
            "Paperless Billing",
            [
                "Select...",
                "Yes",
                "No"
            ]
        )

    with col3:

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Select...",
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )


    col1, col2, col3 = st.columns(3)

    with col1:

        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            value=70.0,
            step=1.0
        )

    with col2:

        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            value=1000.0,
            step=10.0
        )

    with col3:

        st.empty()


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    st.markdown("<br>", unsafe_allow_html=True)

    predict_button = st.button(
        " Run Churn Prediction",
        use_container_width=True
    )


# ============================================================
# RIGHT SIDE — RISK PANEL
# ============================================================

with right_column:

    st.markdown(
        '<div class="risk-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="risk-title">CHURN RISK</div>',
        unsafe_allow_html=True
    )

    if not predict_button:

        st.markdown(
            '<div class="risk-number">—</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="risk-description">'
            'Complete the customer information and run the prediction '
            'to calculate the estimated churn probability.'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# VALIDATION + PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # CHECK SELECT FIELDS
    # --------------------------------------------------------

    selections = [
        gender,
        senior_citizen,
        partner,
        dependents,
        phone_service,
        multiple_lines,
        internet_service,
        online_security,
        online_backup,
        device_protection,
        tech_support,
        streaming_tv,
        streaming_movies,
        contract,
        paperless_billing,
        payment_method
    ]

    if "Select..." in selections:

        st.warning(
            " Please select an option for every dropdown field before running the prediction."
        )

        st.stop()


    # --------------------------------------------------------
    # CONVERT SENIOR CITIZEN TO INTEGER
    # --------------------------------------------------------

    senior_citizen = int(senior_citizen)


    # --------------------------------------------------------
    # CREATE CUSTOMER DATAFRAME
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # ONE-HOT ENCODING
    # --------------------------------------------------------

    customer_encoded = pd.get_dummies(
        customer,
        drop_first=True
    )


    # --------------------------------------------------------
    # MATCH TRAINING COLUMNS
    # --------------------------------------------------------

    customer_encoded = customer_encoded.reindex(
        columns=columns,
        fill_value=0
    )


    # --------------------------------------------------------
    # SCALE FEATURES
    # --------------------------------------------------------

    customer_scaled = scaler.transform(
        customer_encoded
    )


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(
        customer_scaled
    )[0]


    # --------------------------------------------------------
    # CHURN PROBABILITY
    # --------------------------------------------------------

    probability = model.predict_proba(
        customer_scaled
    )[0][1]


    probability_percent = probability * 100


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '<span class="section-number">04</span>'
        'Prediction Result'
        '</div>',
        unsafe_allow_html=True
    )


    result_col1, result_col2 = st.columns([1, 1])


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    with result_col1:

        if prediction == 1:

            st.markdown(
                '<div class="result-high">'
                ' HIGH CHURN RISK<br><br>'
                '<span style="font-weight:400;">'
                'The model predicts that this customer is likely to churn.'
                '</span>'
                '</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="result-low">'
                '✓ LOW CHURN RISK<br><br>'
                '<span style="font-weight:400;">'
                'The model predicts that this customer is likely to stay.'
                '</span>'
                '</div>',
                unsafe_allow_html=True
            )


    # --------------------------------------------------------
    # PROBABILITY
    # --------------------------------------------------------

    with result_col2:

        st.metric(
            "Churn Probability",
            f"{probability_percent:.2f}%"
        )

        st.progress(
            min(int(probability_percent), 100)
        )


    # ========================================================
    # SUMMARY METRICS
    # ========================================================

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Prediction",
            "CHURN" if prediction == 1 else "STAY"
        )


    with col2:

        st.metric(
            "Probability",
            f"{probability_percent:.2f}%"
        )


    with col3:

        st.metric(
            "Risk Level",
            "High" if probability >= 0.5 else "Low"
        )


    st.caption(
        "Prediction generated using the trained Telco Customer Churn machine learning model."
    )
