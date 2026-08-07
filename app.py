# ============================
# Import Libraries
# ============================

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load Models
logistic_model = joblib.load("logistic_model.pkl")
gradient_model = joblib.load("gradient_boosting_model.pkl")

# Load QuantileTransformer
qt = joblib.load("quantile_transformer.pkl")

# ============================
# Project Title
# ============================
st.title("❤️ Heart Disease Prediction")
# ============================
# Project Description
# ============================

st.write("""
This application predicts whether a person is at risk of cardiovascular disease
using Machine Learning based on medical information.
""")
# ============================
# User Input
# ============================

age = st.number_input("Age (Days)", min_value=10000, max_value=30000)

gender = st.selectbox(
    "Gender",
    [1, 2],
    format_func=lambda x: "Female" if x == 1 else "Male"
)

height = st.number_input("Height (cm)", min_value=100, max_value=250)

weight = st.number_input("Weight (kg)", min_value=30, max_value=200)

ap_hi = st.number_input("Systolic Blood Pressure")

ap_lo = st.number_input("Diastolic Blood Pressure")

cholesterol = st.selectbox(
    "Cholesterol",
    [1, 2, 3],
    format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1]
)

gluc = st.selectbox(
    "Glucose",
    [1, 2, 3],
    format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1]
)

smoke = st.selectbox(
    "Smoking",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

alco = st.selectbox(
    "Alcohol Intake",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

active = st.selectbox(
    "Physical Activity",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

# ============================
# Prepare Input Data
# ============================


# Create input DataFrame
input_data = pd.DataFrame([[
    age,
    gender,
    height,
    weight,
    ap_hi,
    ap_lo,
    cholesterol,
    gluc,
    smoke,
    alco,
    active
]], columns=[
    "age",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active"
])


# ============================
# Same preprocessing as training
# ============================

# 1. Convert gender from 1,2 to 0,1
input_data["gender"] = input_data["gender"] - 1


# 2. Apply QuantileTransformer
features_to_transform = [
    "ap_hi",
    "ap_lo",
    "age",
    "height",
    "weight"
]

input_data[features_to_transform] = qt.transform(
    input_data[features_to_transform]
)


# 3. One-Hot Encoding
input_data = pd.get_dummies(
    input_data,
    columns=["cholesterol", "gluc"],
    drop_first=True,
    dtype=int
)


# ============================
# Make sure columns are identical
# to the columns used during training
# ============================

expected_columns = [
    "age",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "smoke",
    "alco",
    "active",
    "cholesterol_2",
    "cholesterol_3",
    "gluc_2",
    "gluc_3"
]

input_data = input_data.reindex(
    columns=expected_columns,
    fill_value=0
)



if st.button("Predict"):

    # Logistic Regression prediction
    logistic_prediction = logistic_model.predict(input_data)

    # Gradient Boosting prediction
    gradient_prediction = gradient_model.predict(input_data)

    # Display Logistic Regression result
    if logistic_prediction[0] == 1:
        st.error("Logistic Regression: ⚠️ High Risk")
    else:
        st.success("Logistic Regression: ✅ Low Risk")

    # Display Gradient Boosting result
    if gradient_prediction[0] == 1:
        st.error("Gradient Boosting: ⚠️ High Risk")
    else:
        st.success("Gradient Boosting: ✅ Low Risk")

    # ============================
    # Final Result (Now properly indented!)
    # ============================
    if logistic_prediction[0] == gradient_prediction[0]:
        if logistic_prediction[0] == 1:
            st.error("🚨 Final Result: High Risk of Cardiovascular Disease")
        else:
            st.success("✅ Final Result: Low Risk of Cardiovascular Disease")
    else:
        st.warning("⚠️ The models gave different predictions")


 # ============================
 # ============================
## ============================
# 📊 Health Data Visualization
# ============================

st.subheader("📊 Patient Health Analysis")

# ----------------------------
# Health Measurements
# ----------------------------

health_data = pd.DataFrame({
    "Health Indicator": [
        "Systolic BP",
        "Diastolic BP",
        "Height",
        "Weight"
    ],
    "Value": [
        ap_hi,
        ap_lo,
        height,
        weight
    ]
})

st.bar_chart(
    health_data.set_index("Health Indicator")
)

# ============================
# 🥧 Cholesterol & Glucose
# ============================

st.subheader("🩸 Cholesterol & Glucose")

col1, col2 = st.columns(2)

# ----------------------------
# Cholesterol
# ----------------------------

with col1:

    cholesterol_labels = {
        1: "Normal",
        2: "Above Normal",
        3: "Well Above Normal"
    }

    cholesterol_status = cholesterol_labels[cholesterol]

    st.metric(
        "Cholesterol",
        cholesterol_status
    )

# ----------------------------
# Glucose
# ----------------------------

with col2:

    glucose_labels = {
        1: "Normal",
        2: "Above Normal",
        3: "Well Above Normal"
    }

    glucose_status = glucose_labels[gluc]

    st.metric(
        "Glucose",
        glucose_status
    )

# ============================
# ❤️ Patient Information
# ============================

st.subheader("❤️ Patient Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Weight", f"{weight} kg")

with col2:
    st.metric("Height", f"{height} cm")

with col3:
    st.metric("Systolic BP", f"{ap_hi}")

with col4:
    st.metric("Diastolic BP", f"{ap_lo}")