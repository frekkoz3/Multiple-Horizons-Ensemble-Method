# DATASETS OVERVIEW

This document provides an overview of the five benchmark time-series datasets analyzed in the paper *Performance metrics for multi-step forecasting measuring win-loss, seasonal variance and forecast stability: an empirical study*. These datasets represent diverse domains, temporal patterns, and levels of predictability.

Original data source, provided by the paper publishers, is available [here](https://zenodo.org/records/6970019).

**General Informations**:
| Dataset     | Domain    | Seasonality            | Noise Level | Key Characteristic              |
| ----------- | --------- | ---------------------- | ----------- | ------------------------------- |
| Electricity | Energy    | Strong daily           | Low         | Individual client load diagrams |
| Traffic     | Transport | Daily / rush-hour      | Moderate    | High-dimensional sensor grid    |
| Volatility  | Finance   | Weak                   | Very high   | Non-seasonal, abrupt changes    |
| Solar       | Energy    | Strong / deterministic | Moderate    | Zero values at night            |
| Wind        | Energy    | Weak / stochastic      | High        | Atmospheric variability         |

---
## Dataset Structural Overview
Some key features extrapolated from the Exploratory Data Analysis we performed:

| Dataset     | # Time Series   | Average Length (per series) | Observation Delta | Notes from EDA                                                         |
|-------------|-----------------|-----------------------------|-------------------|------------------------------------------------------------------------|
| Electricity | 369 clients     | 5957 observations           | 1 hour            | Regular sampling, strong daily and weekly patterns, no missing values. |
| Traffic     | 963 sensors     | 4151 observations           | 1 hour            | Stored as daily slices; strong intraday structure, moderate noise.     |
| Volatility  | 31 indices      | 4898 observations           | 1 day             | Irregular gaps in raw data; high noise, frequent regime shifts.        |
| Solar       | 137 plants      | 52560 observations          | 10 minutes        | Deterministic daily cycle, zero-valued nights, bounded signal.         |
| Wind        | 28 countries    | 10957 observations          | 1 day             | Weak seasonality, high stochastic variability, climate-driven trends.  |



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
* Structure: Original 10 minute delayed observations have been aggregated to hourly observations.
* Technical Notes:
  * Labels indicate the day of the week (1 = Monday, 7 = Sunday).
  * Public holidays and two specific anomalous days (March 2008 and March 2009) have been removed.

---
## REALIZED VOLATILITY

A financial dataset focusing on the daily volatility of 31 international stock indices. It is used to evaluate how models handle noisy, non-seasonal data.

* Content: Aggregated daily realized volatility computed from high-frequency intraday returns.
* Temporal Features: Exhibits weak seasonality and high noise.
* Key Challenges: Frequent abrupt regime changes and lack of regular patterns compared to energy or traffic data.

---
## SOLAR POWER

A generation dataset from photovoltaic plants characterized by clear, deterministic cycles.

* Content: Solar power generation values.
* Temporal Features: Strong daily cycles driven by the sun, along with longer-term seasonal weather variations.
* Key Challenges: Values are strictly zero during nighttime; values during the day can vary greatly due to adverse weather.

---
## WIND POWER

A dataset covering wind power generation across 29 countries, reflecting the stochastic nature of weather.

* Content: Normalized power output expressed as a percentage of total capacity.
* Temporal Features: High short-term variability with a much weaker seasonal structure than solar power.
* Key Challenges: Highly influenced by atmospheric conditions, resulting in frequent and unpredictable fluctuations.