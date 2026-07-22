import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -------------------------
# Load Model and Encoders
# -------------------------

model = joblib.load("credit_scoring_model.pkl")
encoders = joblib.load("label_encoders.pkl")

# -------------------------
# Page Configuration
# -------------------------

st.set_page_config(
    page_title="Credit Scoring System",
    page_icon="🏦",
    layout="centered"
)

# -------------------------
# Sidebar
# -------------------------

st.sidebar.title("🏦 Credit Scoring System")

st.sidebar.info("""
Developed by: Yash Awari

Model: Decision Tree Classifier

Cross Validation Accuracy: 93.10%

Dataset: Credit Score Dataset
""")

# -------------------------
# Title
# -------------------------

st.title("🏦 Credit Scoring Prediction")

st.write(
    "Predict an individual's credit score category based on financial and demographic information."
)

st.markdown("---")

# -------------------------
# User Inputs
# -------------------------

st.subheader("Enter Customer Details")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

gender = st.selectbox(
    "Gender",
    encoders["Gender"].classes_
)

income = st.number_input(
    "Annual Income",
    min_value=0,
    value=50000,
    step=1000
)

education = st.selectbox(
    "Education",
    encoders["Education"].classes_
)

marital_status = st.selectbox(
    "Marital Status",
    encoders["Marital Status"].classes_
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0
)

home = st.selectbox(
    "Home Ownership",
    encoders["Home Ownership"].classes_
)

predict = st.button("Predict Credit Score")

if predict:

    gender_encoded = encoders["Gender"].transform([gender])[0]
    education_encoded = encoders["Education"].transform([education])[0]
    marital_encoded = encoders["Marital Status"].transform([marital_status])[0]
    home_encoded = encoders["Home Ownership"].transform([home])[0]

    input_data = pd.DataFrame({
        "Age":[age],
        "Gender":[gender_encoded],
        "Income":[income],
        "Education":[education_encoded],
        "Marital Status":[marital_encoded],
        "Number of Children":[children],
        "Home Ownership":[home_encoded]
    })

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)

    confidence = np.max(probabilities) * 100

    st.subheader("Prediction Probabilities")

    prob_df = pd.DataFrame({
    "Credit Score": encoders["Credit Score"].classes_,
    "Probability (%)": (probabilities[0] * 100).round(2)
})

    st.dataframe(prob_df, use_container_width=True)

    st.bar_chart(
    prob_df.set_index("Credit Score")
)

    label = encoders["Credit Score"].inverse_transform([prediction])[0]

    st.markdown("---")
    st.subheader("Prediction")

    if label == "High":
        st.success("🟢 High Credit Score")

    elif label == "Average":
        st.warning("🟡 Average Credit Score")

    else:
        st.error("🔴 Low Credit Score")

        st.subheader("Customer Summary")

    st.write(f"**Age:** {age}")
    st.write(f"**Gender:** {gender}")
    st.write(f"**Income:** ₹{income:,}")
    st.write(f"**Education:** {education}")
    st.write(f"**Marital Status:** {marital_status}")
    st.write(f"**Children:** {children}")
    st.write(f"**Home Ownership:** {home}")

st.markdown("---")

st.warning("""
This application is intended for educational purposes only.
The prediction should not be used as a substitute for professional financial assessment.
""")

st.markdown(
    "<center>© 2026 Credit Scoring System | AI & Data Science Project</center>",
    unsafe_allow_html=True
)