
# AI Bird-Strike Risk Forecasting  🇵🇹

A hybrid ML + DL pipeline that predicts the daily probability of a bird-strike event (and expected strike count) for every major Portuguese airport.

* **XGBoost classifier** – 30-minute resolution, trained on 7 cyclic time features + strike count.
* **Seq2Seq RNN** – encoder–decoder LSTM (64 units each) that produces a event-driven forecast.
* **Risk analytics** – per-airport bird-mass factor, probability aggregation, daily risk score and 30-minute breakdown.




## 1  Methods

### 1.1  XGBoost Machine Learning Model

| item | value |
|------|-------|
| **features** | sin/cos hour, sin/cos DOW, sin/cos month, hour\_norm, strike count |
| **target**   | binary – was there any strike in this half-hour? |
| **grid**     | `max_depth` {3,5,7}, `learning_rate` {0.01,0.1}, `n_estimators` {100,200},<br>`subsample` {0.8,1.0}, `colsample_bytree` {0.8,1.0} |
| **CV**       | 3-fold, ROC-AUC |

Best parameters and fold metrics are logged per airport.


### 1.2  Seq2Seq RNN Forecasting Agent

| part | detail |
|------|--------|
| **encoder** | 48 × 30 min window (24 h) → LSTM(64) |
| **decoder** | LSTM(64) → 120 hourly steps (5 days) |
| **heads**   | *count* (MSE) &nbsp;&nbsp;•&nbsp;&nbsp; *prob* (binary-XE) |
| **inputs**  | past features + XGBoost proba (stacked channel) |
| **training**| early-stopping = 10, batch = 32, epochs ≤ 100 |
| **Heads**| 2, one for ML other for RNN |
| **Neurons**| 130 per head |
| **Number of Parameters**| Approximately 1 Million |




## 2  Risk Calculation

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




## 3  Usage

Precompute metrics & forecasts




## 4  Data

Source: EASA Bird‑Strikes dataset

Species mass: Specific & generic bird masses



## 5  License

Luis Santos: perigeu@gmail.com | luis.santos@iseclisboa.pt
