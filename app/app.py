# app.py
import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt

# Load model and metadata
model = joblib.load('model/best_heart_disease_model.joblib')
meta = json.load(open('model/feature_metadata.json'))

# Default threshold from training
DEFAULT_THRESHOLD = 0.12

# Categorical options
CATEGORICAL_OPTIONS = {
    'Smoking': ['yes', 'no'],
    'AlcoholDrinking': ['yes', 'no'],
    'Stroke': ['yes', 'no'],
    'DiffWalking': ['yes', 'no'],
    'Sex': ['male', 'female'],
    'AgeCategory': ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49',
                    '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+'],
    'Race': ['white', 'black', 'asian', 'american indian/alaskan native',
             'other', 'hispanic'],
    'Diabetic': ['yes', 'no', 'no, borderline diabetes', 'yes (during pregnancy)'],
    'PhysicalActivity': ['yes', 'no'],
    'GenHealth': ['excellent', 'very good', 'good', 'fair', 'poor'],
    'Asthma': ['yes', 'no'],
    'KidneyDisease': ['yes', 'no'],
    'SkinCancer': ['yes', 'no']
}

# Sidebar content
with st.sidebar:
    st.header("About This App")
    st.markdown("""
    - **Model**: XGBoost Classifier
    - **Training Data**: CDC 2020 Behavioral Risk Factor Surveillance System  
    - **Key Features**:
        - 17 health & demographic factors
        - Optimized for early detection  
    - **Default Threshold**: 0.12 (30% recall)
    """)

    st.header("Prediction Threshold")
    threshold = st.slider(
        "Decision threshold (↑ increases precision, ↓ increases sensitivity)",
        min_value=0.01,
        max_value=0.99,
        value=DEFAULT_THRESHOLD,
        step=0.01,
        help="Default threshold optimized for 30% recall"
    )
    st.caption(f"🔧 Current threshold: {threshold:.2f}")

    st.header("Interpretation Guide")
    st.markdown("""
    - **Risk Probability**: Model's confidence (0-100%)  
    - **High Risk**: Probability ≥ threshold  
    - **Threshold Adjustment**:  
        - ↑ Threshold → Fewer false positives  
        - ↓ Threshold → More true positives  
    - **ROC Curve**: Shows model's true positive vs false positive rate  
    """)

    st.warning("⚠️ This tool provides risk estimates, not medical diagnoses. Consult a healthcare professional for medical advice.")

# Main title
st.title('Heart Disease Risk Prediction')
st.markdown("""
This app predicts your risk of heart disease using a machine learning model trained on CDC health data.  
Adjust the threshold in the sidebar to balance sensitivity and specificity.
""")

# Input form
with st.form("prediction_form"):
    st.header("Patient Information")

    # Numerical inputs
    cols = st.columns(4)
    bmi = cols[0].number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)
    physical_health = cols[1].number_input('Physical Health Days (past 30 days)', 0, 30, 0)
    mental_health = cols[2].number_input('Mental Health Days (past 30 days)', 0, 30, 0)
    sleep_time = cols[3].number_input('Sleep Hours (per 24h)', 1.0, 24.0, 7.0)

    # Categorical inputs
    st.subheader("Health History & Behaviors")
    cols = st.columns(3)
    smoking = cols[0].selectbox('Smoker', CATEGORICAL_OPTIONS['Smoking'])
    alcohol = cols[1].selectbox('Heavy Alcohol Consumption', CATEGORICAL_OPTIONS['AlcoholDrinking'])
    stroke = cols[2].selectbox('History of Stroke', CATEGORICAL_OPTIONS['Stroke'])

    cols = st.columns(3)
    diff_walking = cols[0].selectbox('Difficulty Walking', CATEGORICAL_OPTIONS['DiffWalking'])
    sex = cols[1].selectbox('Sex', CATEGORICAL_OPTIONS['Sex'])
    age = cols[2].selectbox('Age Category', CATEGORICAL_OPTIONS['AgeCategory'])

    cols = st.columns(3)
    race = cols[0].selectbox('Race/Ethnicity', CATEGORICAL_OPTIONS['Race'])
    diabetic = cols[1].selectbox('Diabetic Status', CATEGORICAL_OPTIONS['Diabetic'])
    phys_activity = cols[2].selectbox('Physical Activity (past 30 days)', CATEGORICAL_OPTIONS['PhysicalActivity'])

    cols = st.columns(3)
    gen_health = cols[0].selectbox('General Health', CATEGORICAL_OPTIONS['GenHealth'])
    asthma = cols[1].selectbox('Asthma', CATEGORICAL_OPTIONS['Asthma'])
    kidney = cols[2].selectbox('Kidney Disease', CATEGORICAL_OPTIONS['KidneyDisease'])

    skin_cancer = st.selectbox('Skin Cancer', CATEGORICAL_OPTIONS['SkinCancer'])

    submitted = st.form_submit_button("Predict Heart Disease Risk")

# Prediction output
if submitted:
    input_data = {
        'BMI': bmi,
        'PhysicalHealth': physical_health,
        'MentalHealth': mental_health,
        'SleepTime': sleep_time,
        'Smoking': smoking,
        'AlcoholDrinking': alcohol,
        'Stroke': stroke,
        'DiffWalking': diff_walking,
        'Sex': sex,
        'AgeCategory': age,
        'Race': race,
        'Diabetic': diabetic,
        'PhysicalActivity': phys_activity,
        'GenHealth': gen_health,
        'Asthma': asthma,
        'KidneyDisease': kidney,
        'SkinCancer': skin_cancer
    }
    input_df = pd.DataFrame([input_data])

    # Predict probability
    proba = model.predict_proba(input_df)[0][1]

    # Display results
    st.subheader("Prediction Results")
    st.metric("Heart Disease Risk Probability", f"{proba:.1%}")

    prediction = "High Risk" if proba >= threshold else "Low Risk"
    st.metric("Heart Disease Risk Classification", prediction, delta=f"Threshold: {threshold:.2f}")

    st.info(f"🔔 Using threshold = {threshold:.2f}: "
            f"Patients with probability ≥ {threshold:.0%} are classified as High Risk")


    # Show submitted data
    st.subheader("Submitted Patient Data")
    st.dataframe(input_df.style.format("{:.2f}", subset=meta['num_feats']))
