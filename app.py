import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Telco Churn Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- Styling ----------
st.markdown("""
<style>
    .stApp {
        background: #f3f6fb;
        color: #17243a;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(120deg, #102344, #1769aa);
        padding: 30px;
        border-radius: 18px;
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: white;
        margin-bottom: 8px;
        font-size: 2.2rem;
    }

    .hero p {
        color: #e2edff;
        margin-bottom: 0;
        font-size: 1rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #17243a;
        margin: 10px 0 18px 0;
    }

    .result-card {
        padding: 24px;
        border-radius: 16px;
        margin-top: 15px;
        border: 1px solid #dbe4f0;
        background: white;
    }

    .risk-high {
        border-left: 6px solid #dc4545;
    }

    .risk-low {
        border-left: 6px solid #1caa78;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.95rem;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 5px 0;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        background: #1769aa;
        color: white;
        border: none;
        border-radius: 10px;
        min-height: 45px;
        font-weight: 700;
        width: 100%;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background: #10558e;
        color: white;
    }

    [data-testid="stForm"] {
        background: white;
        border: 1px solid #dbe4f0;
        border-radius: 16px;
        padding: 22px;
    }

    [data-testid="stWidgetLabel"] p {
        font-weight: 600;
        color: #243650;
    }

    .footnote {
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 18px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Load model ----------
@st.cache_resource
def load_bundle():
    return joblib.load("churn_model_bundle.pkl")

bundle = load_bundle()

model = bundle["model"]
contract_encoder = bundle["contract_encoder"]
binary_encoders = bundle["binary_encoders"]
target_encoder = bundle["target_encoder"]
onehot_encoder = bundle["onehot_encoder"]
scaler = bundle["scaler"]

ohe_cols = bundle["ohe_cols"]
num_cols = bundle["num_cols"]
binary_cols = bundle["binary_cols"]
feature_columns = bundle["feature_columns"]

# ---------- Header ----------
st.markdown("""
<div class="hero">
    <h1> Telco Churn Analytics</h1>
    <p>Customer retention insights powered by machine learning</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.05, 0.95], gap="large")

# ---------- Input form ----------
with left:
    st.markdown(
        '<div class="section-title"> Customer Information</div>',
        unsafe_allow_html=True
    )

    st.write("Enter the key details below to estimate churn risk.")

    with st.form("customer_form"):
        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12,
            step=1
        )

        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=1.0
        )

        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No"]
        )

        submitted = st.form_submit_button(
            "  Predict Churn",
            use_container_width=True
        )

# ---------- Prediction result ----------
with right:
    st.markdown(
        '<div class="section-title"> Prediction Result</div>',
        unsafe_allow_html=True
    )

    if not submitted:
        st.markdown("""
        <div class="result-card">
            <h3>Ready to analyze</h3>
            <p>Enter customer details and select
            <b>Predict Churn</b> to view the prediction.</p>
            <p>The dashboard will show the predicted
            outcome and estimated churn probability.</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        total_charges = monthly_charges * tenure

        customer = pd.DataFrame([{
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": tenure,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": internet,
            "Contract": contract,
            "PaperlessBilling": "Yes",
            "OnlineSecurity": (
                "No internet service"
                if internet == "No" else online_security
            ),
            "OnlineBackup": (
                "No internet service" if internet == "No" else "No"
            ),
            "DeviceProtection": (
                "No internet service" if internet == "No" else "No"
            ),
            "TechSupport": (
                "No internet service" if internet == "No" else "No"
            ),
            "StreamingTV": (
                "No internet service" if internet == "No" else "No"
            ),
            "StreamingMovies": (
                "No internet service" if internet == "No" else "No"
            ),
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }])

        try:
            customer["Contract"] = contract_encoder.transform(
                customer[["Contract"]]
            ).ravel()

            for col in binary_cols:
                customer[col] = binary_encoders[col].transform(
                    customer[col]
                )

            encoded_values = onehot_encoder.transform(
                customer[ohe_cols]
            )

            encoded_df = pd.DataFrame(
                encoded_values,
                columns=onehot_encoder.get_feature_names_out(ohe_cols),
                index=customer.index
            )

            customer = pd.concat(
                [customer.drop(columns=ohe_cols), encoded_df],
                axis=1
            )

            customer[num_cols] = scaler.transform(
                customer[num_cols]
            )

            customer = customer[feature_columns]

            prediction = model.predict(customer)[0]

            predicted_label = target_encoder.inverse_transform(
                [int(prediction)]
            )[0]

            probabilities = model.predict_proba(customer)[0]
            churn_index = list(target_encoder.classes_).index("Yes")
            churn_probability = float(probabilities[churn_index])

            if predicted_label == "Yes":
                st.markdown(f"""
                <div class="result-card risk-high">
                    <div class="metric-label">PREDICTED OUTCOME</div>
                    <h2> At Risk of Churning</h2>
                    <p>This customer may be likely to leave.
                    Consider reviewing retention options.</p>
                    <div class="metric-label">ESTIMATED CHURN PROBABILITY</div>
                    <div class="metric-value" style="color:#dc4545">
                        {churn_probability:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card risk-low">
                    <div class="metric-label">PREDICTED OUTCOME</div>
                    <h2> Likely to Stay</h2>
                    <p>This customer is predicted to continue
                    the subscription.</p>
                    <div class="metric-label">ESTIMATED CHURN PROBABILITY</div>
                    <div class="metric-value" style="color:#1caa78">
                        {churn_probability:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.write("**Churn probability indicator**")
            st.progress(churn_probability)

            st.caption(
                "Estimate only. Unentered customer fields use "
                "default assumptions, which can affect the result."
            )

        except Exception as e:
            st.error("Could not generate the prediction.")
            st.exception(e)

# ---------- Footer ----------
st.markdown("""
<div class="footnote">
    Telco Churn Analytics · Machine Learning Prediction Demo
</div>
""", unsafe_allow_html=True)
