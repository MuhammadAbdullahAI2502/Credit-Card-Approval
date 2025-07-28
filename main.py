import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, encoders
model = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

st.title("💳 Credit Card Approval Prediction App")
st.markdown("Enter the applicant's information below:")

# Input Fields (based on your CSV columns)
CODE_GENDER = st.selectbox("Gender", ["M", "F"])
FLAG_OWN_CAR = st.selectbox("Owns Car?", ["Y", "N"])
FLAG_OWN_REALTY = st.selectbox("Owns Realty?", ["Y", "N"])
CNT_CHILDREN = st.number_input("Number of Children", 0, 20, step=1)
AMT_INCOME_TOTAL = st.number_input("Total Income", 0.0)
NAME_INCOME_TYPE = st.selectbox("Income Type", ["Working", "Commercial associate", "Pensioner", "State servant", "Student"])
NAME_EDUCATION_TYPE = st.selectbox("Education", ["Higher education", "Secondary / secondary special", "Incomplete higher", "Lower secondary", "Academic degree"])
NAME_FAMILY_STATUS = st.selectbox("Family Status", ["Married", "Single / not married", "Civil marriage", "Widow", "Separated"])
NAME_HOUSING_TYPE = st.selectbox("Housing Type", ["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment", "Co-op apartment"])
DAYS_BIRTH = st.number_input("Days Since Birth (Negative Value)", -25000, -7000, step=1)
DAYS_EMPLOYED = st.number_input("Days Employed (Negative if Employed)", -50000, 0, step=1)
FLAG_MOBIL = st.selectbox("Has Mobile?", [0, 1])
FLAG_WORK_PHONE = st.selectbox("Has Work Phone?", [0, 1])
FLAG_PHONE = st.selectbox("Has Phone?", [0, 1])
FLAG_EMAIL = st.selectbox("Has Email?", [0, 1])
OCCUPATION_TYPE = st.selectbox("Occupation", list(encoders["OCCUPATION_TYPE"].classes_))
CNT_FAM_MEMBERS = st.number_input("Family Members", 1.0, 20.0, step=1.0)

# Prepare input dataframe
input_data = pd.DataFrame({
    "CODE_GENDER": [CODE_GENDER],
    "FLAG_OWN_CAR": [FLAG_OWN_CAR],
    "FLAG_OWN_REALTY": [FLAG_OWN_REALTY],
    "CNT_CHILDREN": [CNT_CHILDREN],
    "AMT_INCOME_TOTAL": [AMT_INCOME_TOTAL],
    "NAME_INCOME_TYPE": [NAME_INCOME_TYPE],
    "NAME_EDUCATION_TYPE": [NAME_EDUCATION_TYPE],
    "NAME_FAMILY_STATUS": [NAME_FAMILY_STATUS],
    "NAME_HOUSING_TYPE": [NAME_HOUSING_TYPE],
    "DAYS_BIRTH": [DAYS_BIRTH],
    "DAYS_EMPLOYED": [DAYS_EMPLOYED],
    "FLAG_MOBIL": [FLAG_MOBIL],
    "FLAG_WORK_PHONE": [FLAG_WORK_PHONE],
    "FLAG_PHONE": [FLAG_PHONE],
    "FLAG_EMAIL": [FLAG_EMAIL],
    "OCCUPATION_TYPE": [OCCUPATION_TYPE],
    "CNT_FAM_MEMBERS": [CNT_FAM_MEMBERS]
})

# Encode categorical columns
for col in input_data.select_dtypes(include='object').columns:
    le = encoders[col]
    input_data[col] = le.transform(input_data[col])

# Scale features
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.success("✅ Credit Card Approved")
    else:
        st.error("❌ Credit Card Rejected")
