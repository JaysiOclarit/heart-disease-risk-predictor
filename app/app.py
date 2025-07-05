# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

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

# Initialize session state
if "predicted" not in st.session_state:
    st.session_state["predicted"] = False

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
    - **Feature Importance**: View most influential health factors  
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

    cols = st.columns(4)
    bmi = cols[0].number_input('BMI', 10.0, 50.0, 25.0)
    physical_health = cols[1].number_input('Physical Health Days (past 30 days)', 0, 30, 0)
    mental_health = cols[2].number_input('Mental Health Days (past 30 days)', 0, 30, 0)
    sleep_time = cols[3].number_input('Sleep Hours (per 24h)', 1.0, 24.0, 7.0)

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

    proba = model.predict_proba(input_df)[0][1]

    st.session_state["input_df"] = input_df
    st.session_state["proba"] = proba
    st.session_state["predicted"] = True

# Show results if predicted
if st.session_state.get("predicted"):
    input_df = st.session_state["input_df"]
    proba = st.session_state["proba"]

    st.subheader("Prediction Results")
    st.metric("Heart Disease Risk Probability", f"{proba:.1%}")

    prediction = "High Risk" if proba >= threshold else "Low Risk"
    st.metric("Heart Disease Risk Classification", prediction, delta=f"Threshold: {threshold:.2f}")

    st.info(f"🔔 Using threshold = {threshold:.2f}: Patients with probability ≥ {threshold:.0%} are classified as High Risk")

    st.subheader("Submitted Patient Data")
    st.dataframe(input_df.style.format("{:.2f}", subset=meta['num_feats']))

    show_feats = st.checkbox("Show Most Important Features", key="feat_importance")

    if show_feats:
        st.subheader("Top Health Factors Influencing Risk")

        preprocessor = model.named_steps['pre']
        xgb_model = model.named_steps['clf']

        num_features = meta['num_feats']
        cat_features = meta['cat_feats']

        encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        encoded_cat_features = encoder.get_feature_names_out(cat_features)

        final_feature_names = np.concatenate([num_features, encoded_cat_features])

        assert len(final_feature_names) == len(xgb_model.feature_importances_), "Mismatch between feature names and model importance array"

        importance_df = pd.DataFrame({
            'Feature': final_feature_names,
            'Importance': xgb_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), palette='viridis')
        ax.set_title("Top 10 Most Important Features")
        st.pyplot(fig)

st.markdown("""---""")
st.info("⚠️ **Disclaimer**: This application provides risk estimates for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult your healthcare provider.")

st.markdown("**App created by: Jan Christer Oclarit**")
