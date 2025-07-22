import streamlit as st
import joblib
import pandas as pd

# 🔐 User Authentication Check
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("⚠️ Please login from the sidebar before accessing this page.")
    st.stop()

# 🚀 Page Setup
st.set_page_config(page_title="🎓 Student Score Predictor", layout="centered")
st.title("🎓 MLOps Student App - Version 1.1 🚀")
st.write("Use the sliders and options below to predict your score:")

# 🧠 Load Model & Feature Columns
try:
    model = joblib.load("models/model.pkl")
    columns = joblib.load("models/columns.pkl")  # Expected feature columns
except Exception as e:
    st.error(f"❌ Failed to load model or columns: {e}")
    st.stop()

# 📋 Input Fields
study_time = st.slider("📚 Study Time (hours)", 0, 10)
attendance = st.slider("✅ Attendance (%)", 50, 100)
gender = st.selectbox("👤 Gender", ["Male", "Female"])

# 🧠 One-hot encode gender
gender_Male = 1 if gender == "Male" else 0

# 🧾 Create input dataframe
input_data = pd.DataFrame([[study_time, attendance, gender_Male]],
                          columns=["study_time", "attendance", "gender_Male"])

# 🔧 Add any missing columns to match model input schema
for col in columns:
    if col not in input_data.columns:
        input_data[col] = 0

# 📐 Reorder columns to match model
input_data = input_data[columns]

# 🎯 Predict Button
if st.button("🎯 Predict Score"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"📊 Predicted Score: **{prediction:.2f}**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
