import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Feature list (excluding dropped columns)
feature_names = [
    'Age', 'Gender', 'Country', 'self_employed', 'family_history',
    'work_interfere', 'no_employees', 'remote_work', 'tech_company',
    'benefits', 'care_options', 'wellness_program', 'seek_help',
    'anonymity', 'leave', 'mental_health_consequence',
    'phys_health_consequence', 'coworkers', 'supervisor',
    'mental_health_interview', 'phys_health_interview',
    'mental_vs_physical', 'obs_consequence'
]

# User input UI
st.title("Mental Health Risk Predictor")

age = st.slider("Age", 18, 100, 30)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
self_employed = st.selectbox("Self-employed", ["Yes", "No"])
family_history = st.selectbox("Family history of mental illness", ["Yes", "No"])
work_interfere = st.selectbox("Work interference", ["Often", "Rarely", "Never", "Sometimes", "Don't know"])
no_employees = st.selectbox("Company size", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
remote_work = st.selectbox("Remote work", ["Yes", "No"])
tech_company = st.selectbox("Tech company", ["Yes", "No"])
benefits = st.selectbox("Mental health benefits", ["Yes", "No", "Don't know"])
care_options = st.selectbox("Care options available", ["Yes", "No", "Not sure"])
wellness_program = st.selectbox("Wellness program", ["Yes", "No", "Don't know"])
seek_help = st.selectbox("Encouraged to seek help", ["Yes", "No", "Don't know"])
anonymity = st.selectbox("Anonymity protected", ["Yes", "No", "Don't know"])
leave = st.selectbox("Ease of medical leave", ["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"])
mental_health_consequence = st.selectbox("Mental health consequence at work", ["Yes", "No", "Maybe"])
phys_health_consequence = st.selectbox("Physical health consequence at work", ["Yes", "No", "Maybe"])
coworkers = st.selectbox("Discuss with coworkers", ["Yes", "No", "Some of them"])
supervisor = st.selectbox("Discuss with supervisor", ["Yes", "No", "Some of them"])
mental_health_interview = st.selectbox("Mental health in interview", ["Yes", "No", "Maybe"])
phys_health_interview = st.selectbox("Physical health in interview", ["Yes", "No", "Maybe"])
mental_vs_physical = st.selectbox("Which is more important", ["Don't know", "Yes", "No"])
obs_consequence = st.selectbox("Observed consequence for others", ["Yes", "No"])

# Convert input into DataFrame
user_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'self_employed': [self_employed],
    'family_history': [family_history],
    'work_interfere': [work_interfere],
    'no_employees': [no_employees],
    'remote_work': [remote_work],
    'tech_company': [tech_company],
    'benefits': [benefits],
    'care_options': [care_options],
    'wellness_program': [wellness_program],
    'seek_help': [seek_help],
    'anonymity': [anonymity],
    'leave': [leave],
    'mental_health_consequence': [mental_health_consequence],
    'phys_health_consequence': [phys_health_consequence],
    'coworkers': [coworkers],
    'supervisor': [supervisor],
    'mental_health_interview': [mental_health_interview],
    'phys_health_interview': [phys_health_interview],
    'mental_vs_physical': [mental_vs_physical],
    'obs_consequence': [obs_consequence]
})

# Encode categorical values using the same encoders used in training
# Load label encoders if saved or rebuild them using your preprocessing script
from sklearn.preprocessing import LabelEncoder
encoders = joblib.load('label_encoders.pkl')  # Save these during preprocessing

for col in user_data.columns:
    if col in encoders:
        le = encoders[col]
        user_data[col] = le.transform(user_data[col])

# Scale
user_scaled = scaler.transform(user_data)

# Predict
if st.button("Predict Risk"):
    prediction = model.predict(user_scaled)[0]
    if prediction == 1:
        st.error("The worker is likely to require mental health treatment.")
    else:
        st.success(" The worker is not likely to require mental health treatment.")
