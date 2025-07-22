import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from pages.login import login

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    login()
    st.stop()

st.set_page_config(page_title="📉 Drift Monitoring", layout="centered")
st.title("📉 Model Drift Monitoring Dashboard")

# Path to real log file
log_path = "logs/metrics_log.csv"

# Check if log file exists
if not os.path.exists(log_path):
    st.warning("⚠️ No log data found yet. Please train the model first.")
    st.stop()

# Load logs
logs_df = pd.read_csv(log_path)

# Convert timestamp and create a run counter
logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"])
logs_df["run"] = range(1, len(logs_df) + 1)

# Show table
st.subheader("🧾 Recent Training Runs")
st.dataframe(logs_df.tail(5), use_container_width=True)

# Drift alert if R² drops
latest_r2 = logs_df.iloc[-1]["r2_score"]
if latest_r2 < 0.75:
    st.error(f"⚠️ Drift Detected! R² has dropped to {latest_r2}")
else:
    st.success(f"✅ Latest R² is {latest_r2} — No drift detected.")

# Plot R² Score over runs
st.subheader("📊 R² Score Over Time")
fig, ax = plt.subplots()
ax.plot(logs_df["run"], logs_df["r2_score"],
        marker="o", color="blue", label="R² Score")
ax.set_xlabel("Training Run")
ax.set_ylabel("R² Score")
ax.set_title("R² Trend")
ax.grid(True)
st.pyplot(fig)

# Plot MSE over runs
st.subheader("📊 MSE Over Time")
fig2, ax2 = plt.subplots()
ax2.plot(logs_df["run"], logs_df["mse"], marker="x", color="red", label="MSE")
ax2.set_xlabel("Training Run")
ax2.set_ylabel("MSE")
ax2.set_title("MSE Trend")
ax2.grid(True)
st.pyplot(fig2)

# Optional: Show entire log
with st.expander("📁 Show Full Log"):
    st.dataframe(logs_df, use_container_width=True)
