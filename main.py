import streamlit as st
import pandas as pd
import joblib

# --- Load saved files ---
model = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

st.title("💳 Credit Card Approval Prediction App")
st.markdown("Enter the applicant's information below:")

# --- Input Fields ---
CODE_GENDER = st.selectbox("Gender", encoders.get("CODE_GENDER").classes_ if "CODE_GENDER" in encoders else ["M", "F"])
FLAG_OWN_CAR = st.selectbox("Owns Car?", encoders.get("FLAG_OWN_CAR").classes_ if "FLAG_OWN_CAR" in encoders else ["Y", "N"])
FLAG_OWN_REALTY = st.selectbox("Owns Realty?", encoders.get("FLAG_OWN_REALTY").classes_ if "FLAG_OWN_REALTY" in encoders else ["Y", "N"])
CNT_CHILDREN = st.number_input("Number of Children", 0, 20, step=1)
AMT_INCOME_TOTAL = st.number_input("Total Income", 0.0)
NAME_INCOME_TYPE = st.selectbox("Income Type", encoders.get("NAME_INCOME_TYPE").classes_ if "NAME_INCOME_TYPE" in encoders else ["Working", "State servant"])
NAME_EDUCATION_TYPE = st.selectbox("Education", encoders.get("NAME_EDUCATION_TYPE").classes_ if "NAME_EDUCATION_TYPE" in encoders else ["Higher education", "Secondary"])
NAME_FAMILY_STATUS = st.selectbox("Family Status", encoders.get("NAME_FAMILY_STATUS").classes_ if "NAME_FAMILY_STATUS" in encoders else ["Married", "Single"])
NAME_HOUSING_TYPE = st.selectbox("Housing Type", encoders.get("NAME_HOUSING_TYPE").classes_ if "NAME_HOUSING_TYPE" in encoders else ["House / apartment", "Rented"])
DAYS_BIRTH = st.number_input("Days Since Birth (Negative Value)", -25000, -7000, step=1)
DAYS_EMPLOYED = st.number_input("Days Employed (Negative if Employed)", -50000, 0, step=1)
FLAG_MOBIL = st.selectbox("Has Mobile?", [0, 1])
FLAG_WORK_PHONE = st.selectbox("Has Work Phone?", [0, 1])
FLAG_PHONE = st.selectbox("Has Phone?", [0, 1])
FLAG_EMAIL = st.selectbox("Has Email?", [0, 1])
OCCUPATION_TYPE = st.selectbox("Occupation", encoders.get("OCCUPATION_TYPE").classes_ if "OCCUPATION_TYPE" in encoders else ["Laborers", "Managers"])
CNT_FAM_MEMBERS = st.number_input("Family Members", 1.0, 20.0, step=1.0)

# --- Prepare input dataframe ---
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

# --- Encode categorical columns ---
for col in input_data.columns:
    if col in encoders:  # only encode if encoder available
        input_data[col] = encoders[col].astype(str)  # make sure it's string before encoding
        input_data[col] = encoders[col].transform(input_data[col])


# --- Scale features ---
input_scaled = scaler.transform(input_data)

# --- Predict ---
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.success("✅ Credit Card Approved")
    else:
        st.error("❌ Credit Card Rejected")
