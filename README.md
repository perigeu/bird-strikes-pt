# bird-strikes-pt

Bird‑Strike Risk Forecasting

This repository implements a hybrid machine learning and deep learning pipeline to forecast bird‑strike risk. It includes:

- XGBoost classification on 30‑minute time bins

- Seq2Seq RNN (encoder–decoder LSTM) for dynamic, event‑driven forecasts

- Generation of risk intervals with probabilities and risk scores



1. Methods

1.1 XGBoost Classifier

Features: 7 cyclical time features (hour, day‑of‑week, month), normalized strike count

Target: binary indicator of any strike in a 30‑min bin

Hyperparameter Grid:

max_depth: [3, 5, 7]

learning_rate: [0.01, 0.1]

n_estimators: [100, 200]

subsample: [0.8, 1.0]

colsample_bytree: [0.8, 1.0]

CV: 3‑fold, optimizing ROC AUC

Best params logged per airport



1.2 Seq2Seq RNN

Architecture:

Encoder: LSTM(64) over a sliding window of 48 half‑hour steps (24 h history)

Decoder: LSTM(64) producing event‑driven forecasts over a dynamic horizon

Two heads:

Count: TimeDistributed(Dense(1), activation=linear)

Probability: TimeDistributed(Dense(1), activation=sigmoid)

Inputs:

Past features + XGBoost probability stacked as extra channel

Zero‑initialized decoder input

Losses:

Count head: MSE

Prob head: binary cross‑entropy

Training:

EarlyStopping on validation loss (patience = 10)

Batch size = 32, epochs ≤ 100



2. Risk Calculation

For each forecast interval:

risk = predicted_strike_count × cumulative_probability × mass_factor

predicted_strike_count: rounded number of strikes in interval

cumulative_probability: sum of half‑hour probabilities until interval end

mass_factor: average bird mass (kg), thresholded at 1.8 kg

Risk Levels

Risk Score

Level

Physical Meaning

Risk < 2.0 - Low - Minimal impact into the aircraft safety

Risk 2.0 – 5.0 - Medium - Elevated risk and safety — consider increased monitoring

Risk > 5.0 - High - Significant risk and safety — implement active mitigation

Intervals are reported until 95% cumulative probability of an event.


3. Usage

Precompute metrics & forecasts: python ML_DL_R12_PT.py

Serve UI: streamlit run pt_ui4.py

View forecasts & download CSV/JSON


4. Data

Source: EASA Bird‑Strikes dataset

Species mass: Specific & generic bird masses



5. License

Luis Santos
