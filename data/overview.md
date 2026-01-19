# DATASETS OVERVIEW

This document provides an overview of the five benchmark time-series datasets analyzed in the paper *Performance metrics for multi-step forecasting measuring win-loss, seasonal variance and forecast stability: an empirical study*. These datasets represent diverse domains, temporal patterns, and levels of predictability.

Original data source, provided by the paper publishers, is available [here](https://zenodo.org/records/6970019).

| Dataset     | Domain    | Seasonality            | Noise Level | Key Characteristic              |
| ----------- | --------- | ---------------------- | ----------- | ------------------------------- |
| Electricity | Energy    | Strong daily           | Low         | Individual client load diagrams |
| Traffic     | Transport | Daily / rush-hour      | Moderate    | High-dimensional sensor grid    |
| Volatility  | Finance   | Weak                   | Very high   | Non-seasonal, abrupt changes    |
| Solar       | Energy    | Strong / deterministic | Moderate    | Zero values at night            |
| Wind        | Energy    | Weak / stochastic      | High        | Atmospheric variability         |

---
## ELECTRICITY (Electricity Load Diagrams)

This dataset tracks the electricity consumption of hundreds of clients, providing a clear example of human-driven daily cycles.

* Content: Consumption values in kW for individual clients.
* Temporal Features: Strong daily seasonality.
* Technical Notes:
  * To obtain kWh, values must be divided by 4.
  * There are no missing values in this dataset.
  * Timestamps follow Portuguese local time.
  * Daylight Saving Time handling: In March (23-hour day), the 1:00-2:00 AM window is set to zero. In October (25-hour day), that window aggregates two hours of consumption.
  * For clients added after 2011, consumption prior to their join date is recorded as zero.

---
## TRAFFIC (PEMS-SF)

A high-dimensional dataset representing traffic occupancy rates from freeway detectors in the San Francisco Bay Area.

* Content: Occupancy values ranging from 0 to 1, sampled every 10 minutes.
* Structure: 440 daily multivariate time series.
  * 963 sensors (dimensions) operating consistently.
  * 144 time steps per day (6 per hour x 24 hours).
* Data Splits: 263 training instances and 173 test instances.
* Technical Notes:
  * Files store daily series as Matlab-formatted matrices (963 rows x 144 columns).
  * Labels indicate the day of the week (1 = Monday, 7 = Sunday).
  * Public holidays and two specific anomalous days (March 2008 and March 2009) have been removed.

---
## REALIZED VOLATILITY

A financial dataset focusing on the daily volatility of 31 international stock indices. It is used to evaluate how models handle noisy, non-seasonal data.

* Content: Aggregated daily realized volatility computed from high-frequency intraday returns.
* Temporal Features: Exhibits weak seasonality and high noise.
* Key Challenges: Frequent abrupt regime changes and lack of regular patterns compared to energy or traffic data.
* Purpose: Benchmarking model robustness in unpredictable financial environments.

---
## SOLAR POWER

A generation dataset from photovoltaic plants characterized by clear, deterministic cycles.

* Content: Solar power generation values (often normalized).
* Temporal Features: Strong daily cycles driven by the sun, along with longer-term seasonal weather variations.
* Key Challenges: Values are strictly zero during nighttime, creating sharp regime transitions and bounded outputs.
* Purpose: Assessing forecasting performance under physical constraints and predictable cycles.

---
## WIND POWER

A dataset covering wind power generation across 29 countries, reflecting the stochastic nature of weather.

* Content: Normalized power output expressed as a percentage of total capacity.
* Temporal Features: High short-term variability with a much weaker seasonal structure than solar power.
* Key Challenges: Highly influenced by atmospheric conditions, resulting in frequent and unpredictable fluctuations.
* Purpose: Evaluating forecast stability and the ability of models to adapt to highly volatile, non-periodic environments.