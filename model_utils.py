import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os


# -----------------------
# Sliding window
# -----------------------
def create_sequences(X, y, time_steps=20):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


# -----------------------
# Baseline Forecast (Naive)
# -----------------------
def baseline_forecast(df, target_col):
    y_true = df[target_col].values[1:]
    preds = df[target_col].shift(1).dropna().values
    return preds, y_true


# -----------------------
# Train + Save LSTM
# -----------------------
def train_lstm_model(df, target_col, save_dir="."):
    # Drop datetime if present
    X = df.drop(columns=[target_col])
    y = df[[target_col]]

    datetime_cols = X.select_dtypes(include=["datetime64[ns]"]).columns
    if len(datetime_cols) > 0:
        print(f"Dropping datetime columns: {list(datetime_cols)}")
        X = X.drop(columns=datetime_cols)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X.values)
    y_scaled = scaler_y.fit_transform(y.values)

    split = int(len(df) * 0.8)
    X_train, X_val = X_scaled[:split], X_scaled[split:]
    y_train, y_val = y_scaled[:split], y_scaled[split:]

    TIME_STEPS = 20
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, TIME_STEPS)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, TIME_STEPS)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(TIME_STEPS, X_train_seq.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=20,
        batch_size=32,
        verbose=1
    )

    # ---------------- Save Model + Scalers ----------------
    model_path = os.path.join(save_dir, "lstm_model.keras")
    scaler_X_path = os.path.join(save_dir, "scaler_X.pkl")
    scaler_y_path = os.path.join(save_dir, "scaler_y.pkl")

    model.save(model_path)  # SavedModel format (recommended)
    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_y, scaler_y_path)

    # ---------------- Evaluate ----------------
    y_pred_scaled = model.predict(X_val_seq)
    y_val_inv = scaler_y.inverse_transform(y_val_seq.reshape(-1, 1))
    y_pred_inv = scaler_y.inverse_transform(y_pred_scaled)

    results = {
        "RMSE": float(np.sqrt(mean_squared_error(y_val_inv, y_pred_inv))),
        "MAE": float(mean_absolute_error(y_val_inv, y_pred_inv)),
        "MAPE": float(np.mean(np.abs((y_val_inv - y_pred_inv) / y_val_inv)) * 100),
        "R²": float(r2_score(y_val_inv, y_pred_inv))
    }

    return model, scaler_X, scaler_y, history, results


# -----------------------
# Forecast with LSTM
# -----------------------
def lstm_forecast(model, scaler_X, scaler_y, df, target_col):
    X = df.drop(columns=[target_col])
    y = df[[target_col]]

    datetime_cols = X.select_dtypes(include=["datetime64[ns]"]).columns
    if len(datetime_cols) > 0:
        X = X.drop(columns=datetime_cols)

    X_scaled = scaler_X.transform(X.values)
    y_scaled = scaler_y.transform(y.values)

    TIME_STEPS = 20
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, TIME_STEPS)

    y_pred_scaled = model.predict(X_seq)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_seq.reshape(-1, 1))

    return y_pred.flatten(), y_true.flatten()


# -----------------------
# Evaluation Utility
# -----------------------
def evaluate_model(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "R²": float(r2_score(y_true, y_pred))
    }
