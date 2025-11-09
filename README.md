# ⚡ SmartGridGuard

### 🧠 AI-Powered Short-Term Energy Load Forecasting & Anomaly Detection

**Authors:**
👩‍💻 *Sristi Saha*
👩‍💻 *Sahana Samanta*

---

## 🌍 Inspiration

Buildings account for nearly **one-third of global energy use and emissions**. As the world moves toward decarbonization, **accurate short-term energy load forecasting** is essential for balancing demand, integrating renewables, and preventing grid instability.

We were inspired by the idea of applying **AI-based forecasting and anomaly detection** to electricity consumption — helping improve **energy efficiency, reliability, and sustainability**. That vision led us to create **SmartGridGuard**.

---

## ⚙️ What It Does

**SmartGridGuard** is an AI-driven system prototype for smart grids and building energy management that:

1. Uses **LSTM deep learning models** to forecast short-term electricity demand.
2. Applies **contextual anomaly detection** using *Isolation Forest* on forecast residuals and calendar-based features (day of week, weekend, holiday).
3. Detects **abnormal load patterns** such as sudden spikes, drops, or irregular consumption — potential signs of inefficiencies or faults.
4. Provides a **Streamlit dashboard** for real-time visualization and interaction:

   * 📈 *Forecast vs Actual Demand*
   * 🚨 *Highlighted Anomalies in Real Time*

> **In essence:** Predict → Detect → Visualize → Act

---

## 🏗️ How We Built It

### 1. Dataset & Preprocessing

* Used the **Electricity Load Forecasting Dataset** from Kaggle.
* Retained **calendar features** (day-of-week, weekend, holiday) as contextual variables.
* Normalized data and converted it into **time-series sequences** for deep learning models.

### 2. Forecasting with LSTM

* Trained a **Long Short-Term Memory (LSTM)** model to predict short-term demand.
* Tuned model using **dropout** and validation splits to improve generalization.
* Evaluated with **RMSE, MAE, MAPE, and R²** metrics.

### 3. Anomaly Detection

* Computed **residuals** (actual − predicted) after forecasting.
* Combined residuals with contextual features and passed them into an **Isolation Forest** model.
* Used **score percentile-based thresholding** instead of fixed contamination for adaptive anomaly detection.

### 4. Visualization & Deployment

* Built an **interactive Streamlit dashboard** to run the entire pipeline:

  * Upload dataset
  * Forecast with LSTM
  * Apply anomaly detection
  * Visualize actual vs forecast and flagged anomalies
* Integrated with the **default Kaggle dataset** for quick demos and evaluation.

---

## 🧩 Challenges We Faced

1. **Data Quality & Variability:** Handling seasonal and noisy data.
2. **False Positives:** Legitimate peaks (e.g., holidays) misclassified as anomalies → solved via contextual features.
3. **Overfitting:** Used dropout, early stopping, and validation splits.
4. **Dynamic Thresholding:** Switched to percentile-based anomaly scoring for flexibility.
5. **Integration with Streamlit:** Ensuring smooth real-time visualization pipeline.

---

## 🏆 Accomplishments

✅ Built a **complete AI pipeline** — from data preprocessing to anomaly visualization.
✅ Achieved **accurate short-term load forecasts** using LSTM.
✅ Designed a **robust hybrid anomaly detector** using contextual features.
✅ Developed an **interactive Streamlit app** supporting retraining and visualization.
✅ Created a **deployable system** for real-world smart grid monitoring.

---

## 💡 What We Learned

* How to **integrate deep learning (LSTM)** with **classical ML (Isolation Forest)** for time-series data.
* The **importance of contextual features** — not every spike is an anomaly.
* **Streamlit** enables rapid, effective ML visualization.
* **Interpretability** is key for real-world trust in ML systems.
* Correct handling of **scaling, windowing, and inverse transformations** is crucial in forecasting.

---

## 🚀 What’s Next for SmartGridGuard

🔹 **Expand Datasets:** Apply to commercial, residential, and industrial buildings.

🔹 **Adopt TS Foundation Models (TSFM):** Use pre-trained transfer learning models.

🔹 **Real-Time Deployment:** Stream and monitor live data continuously.

🔹 **Explainable AI (XAI):** Integrate SHAP for interpretability of anomaly causes.

🔹 **IoT Integration:** Connect with smart meters and energy management systems.

🔹 **Actionable Insights:** Move beyond flagging — suggest next steps (e.g., HVAC tuning, load scheduling).

---

## 📊 Tech Stack

| Component              | Technology                                    |
| ---------------------- | --------------------------------------------- |
| **Forecasting Model**  | LSTM (Keras, TensorFlow)                      |
| **Anomaly Detection**  | Isolation Forest (scikit-learn)               |
| **Frontend Dashboard** | Streamlit                                     |
| **Data Handling**      | Pandas, NumPy                                 |
| **Visualization**      | Matplotlib, Plotly                            |
| **Dataset Source**     | Kaggle - Electricity Load Forecasting Dataset |

