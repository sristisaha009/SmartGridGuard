import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import os

from model_utils import lstm_forecast, train_lstm_model
from anomaly_utils import detect_anomalies

# -----------------------
# Config
# -----------------------
MODEL_PATH = "lstm_model.keras"
SCALER_X_PATH = "scaler_X.pkl"
SCALER_Y_PATH = "scaler_y.pkl"
DATA_PATH = "elf_dataset.xlsx"
TARGET_COL = "DEMAND"

# -----------------------
# Load model + scalers
# -----------------------
@st.cache_resource
def load_model_and_scalers():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_X_PATH) and os.path.exists(SCALER_Y_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler_X = joblib.load(SCALER_X_PATH)
        scaler_y = joblib.load(SCALER_Y_PATH)
        return model, scaler_X, scaler_y
    else:
        return None, None, None


# -----------------------
# Load dataset
# -----------------------
@st.cache_data
def load_data():
    return pd.read_excel(DATA_PATH)


# -----------------------
# Streamlit App
# -----------------------
st.set_page_config(page_title="SmartGridGuard", layout="wide")
st.title("⚡ SmartGridGuard – Energy Forecasting & Anomaly Detection")

# File uploader for retraining
uploaded_file = st.file_uploader("Upload new dataset (.xlsx) to retrain model", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    # Retrain model on new dataset
    st.info("Training new LSTM model with uploaded dataset...")
    model, scaler_X, scaler_y, history, results = train_lstm_model(df, TARGET_COL)

    # Save artifacts
    model.save(MODEL_PATH)
    joblib.dump(scaler_X, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)

    st.success("✅ Model retrained and saved successfully!")
else:
    df = load_data()

# Show dataset preview
st.markdown("### 📂 Loaded Dataset (Preview)")
st.dataframe(df.head())

# Load pretrained model + scalers
model, scaler_X, scaler_y = load_model_and_scalers()
if model is None:
    st.error("No pre-trained model found. Please upload dataset to train.")
    st.stop()

# -----------------------
# Forecasting
# -----------------------
lstm_preds, y_true_lstm = lstm_forecast(model, scaler_X, scaler_y, df, TARGET_COL)

# Detect anomalies using score percentile (top 0.5% most anomalous points)
anomalies, iso_model = detect_anomalies(
    df, 
    df[TARGET_COL], 
    lstm_preds, 
    contamination=None,        # disable fixed proportion
    score_percentile=98.5      # flag only extreme spikes
)







# -----------------------
# Plots
# -----------------------
st.subheader("📊 Forecasting & Anomaly Detection")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**LSTM Forecast vs Actual**")
    fig, ax = plt.subplots(figsize=(6,3))   # smaller graph
    ax.plot(y_true_lstm, label="Actual")
    ax.plot(lstm_preds, label="LSTM")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.markdown("**Anomaly Detection Results (Isolation Forest)**")
    fig2, ax2 = plt.subplots(figsize=(6,3))
    ax2.plot(df[TARGET_COL].values, label="Actual", alpha=0.8)
    ax2.scatter(anomalies["index"], anomalies["actual"], 
                color="red", label="Anomaly", s=20)
    ax2.legend()
    st.pyplot(fig2)
