SmartGridGuard: AI-Powered Short-Term Energy Load Forecasting and Anomaly Detection
Authors :
Sahana Samanta, Sristi Saha 

Inspiration :
Buildings account for nearly one-third of global energy use and emissions. As the world pushes toward decarbonization, accurate short-term energy load forecasting becomes crucial for balancing demand, integrating renewable energy, and preventing grid instability.
We were inspired by the idea of applying AI-based forecasting and anomaly detection to electricity consumption. The thought that our system could contribute — even in a small way — toward energy efficiency, reliability, and sustainability motivated us to build SmartGridGuard.

What It Does :
1. SmartGridGuard is an AI-driven forecasting and anomaly detection system designed for smart grids and building energy management.
2. Uses LSTM deep learning models to forecast short-term electricity demand.
3. Applies contextual anomaly detection using Isolation Forest on top of forecast errors and calendar features (day of week, weekends, holidays).
4. Identifies abnormal load patterns such as spikes, drops, or unexpected behavior that may indicate equipment faults, inefficiencies, or unusual consumption.
5. Provides a Streamlit dashboard for interactive visualization:
  a. Forecast vs. Actual Demand
  b. Highlighted Anomalies in real-time
In essence: Predict → Detect → Visualize → Act

How We Built It :
We built SmartGridGuard as an end-to-end system that takes raw electricity demand data, forecasts future load, and flags anomalies for grid reliability.
1. Dataset & Preprocessing
	a. We used the Electricity Load Forecasting dataset from Kaggle as the primary source of historical demand data.
	b. This dataset already contained calendar features such as day-of-week, weekend, and holiday indicators, which were retained as contextual variables.
	c. Data was normalized and converted into time-series sequences suitable for deep learning models.
2. Forecasting with LSTM
	a. We trained a Long Short-Term Memory (LSTM) model to predict short-term electricity demand.
	b. The model was tuned to balance accuracy and generalization, using techniques like dropout to reduce overfitting.
	c. Evaluation was done using RMSE, MAE, MAPE, and R².
3. Anomaly Detection
	a. After generating forecasts, we computed the residuals (difference between actual and predicted values).
	b. These residuals, along with contextual features, were passed into an Isolation Forest.
	c. Instead of a fixed contamination rate, we applied a score-based thresholding method: anomalies were flagged if their anomaly scores exceeded a chosen percentile cutoff.
4. Visualization & Deployment
	a. The full pipeline was visualised using a Streamlit dashboard for interactive use.
	b. View the default Kaggle dataset results already integrated into the app (used for our evaluation).
	c. Upload new datasets.
	d. Run the LSTM model for forecasting.
	e. Apply anomaly detection.
	f. View results in real time with intuitive plots (Actual vs Forecast and Flagged Anomalies).

Challenges We Ran Into :
1. Data quality & variability: Energy demand data is noisy and heavily seasonal, making anomaly detection tricky.
2. False positives: Certain demand peaks (e.g., holidays, weekends) looked anomalous at first but were legitimate. We solved this by integrating calendar features (dow_sin/cos, weekend, holiday).
3. Model generalization: Preventing overfitting of the LSTM on historical data required careful tuning (dropout, validation splits).
4. Thresholding anomalies: Isolation Forest sometimes flagged too many/too few anomalies. Using score percentiles instead of fixed contamination gave better flexibility.
5. Integration with Streamlit: Ensuring the pipeline (training → forecasting → anomaly detection → plotting) worked seamlessly in an interactive dashboard took effort.

Accomplishments That We’re Proud Of:
1. Built a fully functional AI pipeline from raw energy data to anomaly visualization.
2. Achieved accurate short-term load forecasting with LSTM.
3. Designed a robust anomaly detection module combining residuals and context features.
4. Developed an interactive Streamlit app that supports retraining, visualization, and user-friendly anomaly insights.
5. Created a system that could be deployed in real-world smart grid monitoring environments.

What We Learned :
1. How to integrate deep learning (LSTM) with classical ML anomaly detection (Isolation Forest) for time series.
2. The importance of contextual features in anomaly detection — not every spike is an anomaly.
3. Streamlit is a powerful tool for rapid prototyping and visualization of ML applications.
4. Real-world ML systems need both predictive performance and interpretability to be trusted.
5. Managing scaling, windowing, and inverse transforms correctly is key in time-series forecasting.

What’s Next for SmartGridGuard :
1. Expand datasets: Apply to diverse building types (commercial, residential, industrial).
2. Incorporate TS Foundation Models (TSFM): Pre-trained models for transfer learning across domains.
3. Real-time deployment: Streaming data integration for continuous monitoring.
4. Explainable AI (XAI): Use SHAP/feature attribution to explain why anomalies are flagged.
5. Integration with IoT & Smart Meters: Deploy directly in energy management systems.
6. Actionable recommendations: Beyond flagging anomalies, suggest what to do (e.g., check HVAC, reschedule equipment usage).
