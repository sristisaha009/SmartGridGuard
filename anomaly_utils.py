import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def detect_anomalies(
    df, y_true, y_pred,
    contamination=None,        # if None, use score threshold instead
    score_percentile=98.5,     # top X% by anomaly score
    rolling_window=24          # for residual baseline (adjust to your data freq)
):
    """
    Contextual anomaly detection using Isolation Forest with error + calendar features.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe containing calendar features (dayofweek, weekend, holiday).
    y_true : array-like
        Actual demand values
    y_pred : array-like
        LSTM forecast values
    contamination : float or None
        Expected anomaly proportion (e.g., 0.02 = 2%). If None, use score_percentile.
    score_percentile : float
        Percentile threshold for anomaly scores (only used if contamination=None).
    rolling_window : int
        Window size for residual smoothing (captures seasonality).

    Returns
    -------
    anomalies : pd.DataFrame
        Rows flagged as anomalies
    iso : IsolationForest
        Trained model
    """

    # Ensure series
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

    errors = (y_true - y_pred).values.reshape(-1, 1)

    # ---- Rolling residuals ----
    rolling_mean = y_true.rolling(rolling_window, min_periods=1).mean()
    residuals = (y_true - rolling_mean) - (y_pred - y_pred.rolling(rolling_window, min_periods=1).mean())

    # ---- Build feature matrix ----
    feature_df = pd.DataFrame({
        "abs_error": np.abs(errors).ravel(),
        "signed_error": errors.ravel(),
        "residual": residuals.fillna(0).values
    })

    # ---- Calendar features ----
    if "dayofweek" in df.columns:
        dow = df["dayofweek"].reset_index(drop=True).iloc[-len(feature_df):].astype(int)
        feature_df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        feature_df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    for col in ["weekend", "holiday"]:
        if col in df.columns:
            feature_df[col] = df[col].reset_index(drop=True).iloc[-len(feature_df):].astype(int)

    # ---- Scale features ----
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df)

    # ---- Train Isolation Forest ----
    iso = IsolationForest(
        contamination=contamination if contamination is not None else 'auto',
        n_estimators=300,
        random_state=42
    )
    iso.fit(X_scaled)

    # anomaly scores (higher = more anomalous)
    scores = -iso.decision_function(X_scaled)

    if contamination is not None:
        labels = iso.predict(X_scaled)
        is_anomaly = labels == -1
    else:
        threshold = np.percentile(scores, score_percentile)
        is_anomaly = scores >= threshold

    # ---- Build output ----
    feature_df["actual"] = y_true
    feature_df["predicted"] = y_pred
    feature_df["score"] = scores
    feature_df["is_anomaly"] = is_anomaly
    feature_df["index"] = feature_df.index

    anomalies = feature_df[feature_df["is_anomaly"]]

    return anomalies, iso
