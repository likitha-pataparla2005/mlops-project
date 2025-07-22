import streamlit as st
import joblib
import pandas as pd

model = joblib.load("models/model.pkl")
columns = joblib.load("models/columns.pkl")  # Load feature names

st.title("🎓 Student Score Predictor")

st.title("🎓 MLOps Student App - Version 1.1 🚀")

study_time = st.slider("Study Time (hours)", 0, 10)
attendance = st.slider("Attendance (%)", 50, 100)
gender = st.selectbox("Gender", ["Male", "Female"])

# Convert gender to one-hot manually
gender_Male = 1 if gender == "Male" else 0

# Create input DataFrame
input_data = pd.DataFrame([[study_time, attendance, gender_Male]], columns=[
                          "study_time", "attendance", "gender_Male"])

# Ensure all expected columns exist
for col in columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder columns to match model
input_data = input_data[columns]

if st.button("Predict Score"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Score: {prediction:.2f}")
