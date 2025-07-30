import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model
model = pickle.load(open('final_loan_model.pkl', 'rb'))

st.title('Loan Approval Prediction')

# Input fields for user data
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input(
    "Applicant Income", min_value=0, step=1000, format="%d")
coapplicant_income = st.number_input(
    "Coapplicant Income", min_value=0, step=1000, format="%d")
loan_amount = st.number_input(
    "Loan Amount", min_value=0, step=1000, format="%d")
loan_amount_term = st.selectbox(
    "Loan Amount Term (in months)", [360, 240, 180, 120, 60])
credit_history = st.selectbox("Credit History", ["Yes", "No"])
property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# Encoding categorical variables
gender_encoded = 1 if gender == "Male" else 0
married_encoded = 1 if married == "Yes" else 0
education_encoded = 1 if education == "Graduate" else 0
self_employed_encoded = 1 if self_employed == "Yes" else 0
credit_history_encoded = 1 if credit_history == "Yes" else 0

# Dependents mapping, treating "3+" as 3
dependents_mapping = {"0": 0, "1": 1, "2": 2, "3+": 3}
dependents_encoded = dependents_mapping[dependents]

property_area_mapping = {"Rural": 0, "Semiurban": 1, "Urban": 2}
property_area_encoded = property_area_mapping[property_area]

# Prepare features as float array
features = np.array([[
    gender_encoded,
    married_encoded,
    dependents_encoded,
    education_encoded,
    self_employed_encoded,
    float(applicant_income),
    float(coapplicant_income),
    float(loan_amount),
    float(loan_amount_term),
    credit_history_encoded,
    property_area_encoded
]], dtype=np.float32)

if st.button("Predict"):
    prediction = model.predict(features.reshape(1, -1))
    result = "Approved" if prediction[0] == 1 else "Not Approved"
    st.subheader(f"Loan Prediction Result: {result}")
