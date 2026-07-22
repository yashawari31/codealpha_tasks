import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessing files
model = joblib.load("heart_disease_rf_model.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("feature_selector.pkl")

# Load dataset for sample patients
df = pd.read_csv("heart.csv")

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction System")
st.write("Enter the patient's details below to predict the likelihood of heart disease.")

# Initialize session state
if "sample" not in st.session_state:
    st.session_state.sample = None

col1, col2 = st.columns(2)

with col1:
    if st.button("🟢 Load Healthy Sample"):
        st.session_state.sample = df[df["target"] == 0].sample(1).iloc[0]

with col2:
    if st.button("🔴 Load Heart Disease Sample"):
        st.session_state.sample = df[df["target"] == 1].sample(1).iloc[0]

sample = st.session_state.sample

age = st.number_input(
    "Age",
    1,
    120,
    int(sample["age"]) if sample is not None else 50
)

sex_options = {
    "Male": 1,
    "Female": 0
}

sex_label = "Male"

if sample is not None:
    sex_label = "Male" if sample["sex"] == 1 else "Female"

sex = st.selectbox(
    "Gender",
    list(sex_options.keys()),
    index=list(sex_options.keys()).index(sex_label)
)

sex = sex_options[sex]


cp_options = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}

cp_reverse = {v: k for k, v in cp_options.items()}

cp = st.selectbox(
    "Chest Pain Type",
    list(cp_options.keys()),
    index=list(cp_options.keys()).index(
        cp_reverse[sample["cp"]] if sample is not None else "Typical Angina"
    )
)

cp = cp_options[cp]


trestbps = st.number_input(
    "Resting Blood Pressure",
    value=int(sample["trestbps"]) if sample is not None else 120
)

chol = st.number_input(
    "Cholesterol",
    value=int(sample["chol"]) if sample is not None else 200
)


fbs_options = {
    "≤120 mg/dL": 0,
    ">120 mg/dL": 1
}

fbs_reverse = {v: k for k, v in fbs_options.items()}

fbs = st.selectbox(
    "Fasting Blood Sugar",
    list(fbs_options.keys()),
    index=list(fbs_options.keys()).index(
        fbs_reverse[sample["fbs"]] if sample is not None else "≤120 mg/dL"
    )
)

fbs = fbs_options[fbs]


restecg_options = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}

rest_reverse = {v: k for k, v in restecg_options.items()}

restecg = st.selectbox(
    "Resting ECG",
    list(restecg_options.keys()),
    index=list(restecg_options.keys()).index(
        rest_reverse[sample["restecg"]] if sample is not None else "Normal"
    )
)

restecg = restecg_options[restecg]


thalach = st.number_input(
    "Maximum Heart Rate",
    value=int(sample["thalach"]) if sample is not None else 150
)


exang_options = {
    "No": 0,
    "Yes": 1
}

exang_reverse = {v: k for k, v in exang_options.items()}

exang = st.selectbox(
    "Exercise Induced Angina",
    list(exang_options.keys()),
    index=list(exang_options.keys()).index(
        exang_reverse[sample["exang"]] if sample is not None else "No"
    )
)

exang = exang_options[exang]


oldpeak = st.number_input(
    "Oldpeak",
    value=float(sample["oldpeak"]) if sample is not None else 1.0
)


slope_options = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

slope_reverse = {v: k for k, v in slope_options.items()}

slope = st.selectbox(
    "Slope",
    list(slope_options.keys()),
    index=list(slope_options.keys()).index(
        slope_reverse[sample["slope"]] if sample is not None else "Upsloping"
    )
)

slope = slope_options[slope]


ca = st.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3, 4],
    index=int(sample["ca"]) if sample is not None else 0
)


thal_options = {
    "Normal": 1,
    "Fixed Defect": 2,
    "Reversible Defect": 3
}

thal_reverse = {v: k for k, v in thal_options.items()}

thal = st.selectbox(
    "Thal",
    list(thal_options.keys()),
    index=list(thal_options.keys()).index(
        thal_reverse[sample["thal"]] if sample is not None else "Normal"
    )
)

thal = thal_options[thal]

if st.button("🔍 Predict"):

    patient = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    st.subheader("Patient Details")

    st.dataframe(patient)

    patient_scaled = scaler.transform(patient)
    patient_selected = selector.transform(patient_scaled)

    prediction = model.predict(patient_selected)
    probability = model.predict_proba(patient_selected)

    disease_prob = probability[0][1] * 100
    healthy_prob = probability[0][0] * 100

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")

    st.subheader("Prediction Confidence")

    st.write(f"❤️ Heart Disease Probability : **{disease_prob:.2f}%**")
    st.progress(int(disease_prob))

    st.write(f"💚 Healthy Probability : **{healthy_prob:.2f}%**")
    st.progress(int(healthy_prob))

    st.subheader("Detailed Probabilities")

    result = pd.DataFrame({
        "Condition": ["Healthy", "Heart Disease"],
        "Probability (%)": [
            round(healthy_prob, 2),
            round(disease_prob, 2)
        ]
    })

    st.table(result)

    st.subheader("Model Information")

    st.info("""
    ✔ Model : Random Forest Classifier

    ✔ Feature Scaling : StandardScaler

    ✔ Feature Selection : SelectKBest (Top 9 Features)

    ✔ Test Accuracy : 98%
    """)

    st.success("Prediction Completed Successfully.")

    if disease_prob >= 80:
      st.error("🔴 High Risk of Heart Disease")
    elif disease_prob >= 50:
       st.warning("🟠 Moderate Risk of Heart Disease")
    else:
       st.success("🟢 Low Risk of Heart Disease")






       st.success("Prediction Completed Successfully.")

       st.warning("""
This application is intended for educational purposes only.
It should not be used as a substitute for professional medical advice.
""")
       
       