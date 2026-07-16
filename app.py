import joblib
import pandas as pd
import streamlit as st
import numpy as np


MODEL_PATH = "best_loan_model.pkl"


@st.cache_resource
def load_model():
    """Load the trained model pipeline."""
    return joblib.load(MODEL_PATH)


st.title("Loan Approval Prediction")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Run python loan_approval_training.py first.")
    st.stop()

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["No", "Yes"])
applicant_income = st.number_input("Applicant Income", min_value=0.0, value=5000.0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0)
loan_amount = st.number_input("Loan Amount", min_value=0.0, value=120.0)
loan_amount_term = st.number_input("Loan Amount Term", min_value=0.0, value=360.0)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

if st.button("Predict"):
    input_data = pd.DataFrame(
        [
            {
                "Gender": gender,
                "Married": married,
                "Dependents": dependents,
                "Education": education,
                "Self_Employed": self_employed,
                "ApplicantIncome": applicant_income,
                "CoapplicantIncome": coapplicant_income,
                "LoanAmount": loan_amount,
                "Loan_Amount_Term": loan_amount_term,
                "Credit_History": credit_history,
                "Property_Area": property_area,
            }
        ]
    )
    input_data["TotalIncome"] = (
        input_data["ApplicantIncome"] + input_data["CoapplicantIncome"]
    )

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")
