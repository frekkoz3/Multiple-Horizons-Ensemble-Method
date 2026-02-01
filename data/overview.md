# DATASETS OVERVIEW

This document provides an overview of the five benchmark time-series datasets analyzed in the paper *Performance metrics for multi-step forecasting measuring win-loss, seasonal variance and forecast stability: an empirical study*. These datasets represent diverse domains, temporal patterns, and levels of predictability.

Original data source, provided by the paper publishers, is available [here](https://zenodo.org/records/6970019).

**General Informations**:
| Dataset     | Domain    | Seasonality            | Noise Level | Key Characteristic              |
| ----------- | --------- | ---------------------- | ----------- | ------------------------------- |
| Electricity | Energy    | Strong daily           | Low         | Individual client load diagrams |
| Traffic     | Transport | Daily                  | Moderate    | High measures during peak times |
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

The Electricity Load Diagrams dataset is collected from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/) and is commonly used as a benchmark for forecasting models. It exhibits high daily seasonality, and each time series varies significantly in magnitude.

* Content: Consumption values in kW for individual clients.
* Temporal Features: Strong daily seasonality.
* Technical Notes:
  * There are no missing values.
  * Timestamps follow Portuguese local time.
  * Daylight Saving Time handling: In March (23-hour day), the 1:00-2:00 AM window is set to zero. In October (25-hour day), that window aggregates two hours of consumption.
  * For clients added after 2011, consumption prior to their join date is recorded as zero.

---
## TRAFFIC (PEMS-SF)

The PEMS-SF dataset is collected from the UCI Machine Learning Repository and is typically used as a benchmark alongside electricity. The dataset contains 15 months worth of daily data that describes the occupancy rate of different car lanes of the San Francisco bay area freeways across time. The dataset features high daily seasonality in addition to peak hour traffic spikes.

* Content: Occupancy values ranging from 0 to 1, sampled every 10 minutes.
* Structure: Original 10 minute delayed observations have been aggregated to hourly observations.
* Technical Notes:
  * Labels indicate the day of the week (1 = Monday, 7 = Sunday).
  * Public holidays and two specific anomalous days (March 2008 and March 2009) have been removed.

---
## REALIZED VOLATILITY

The volatility dataset is collected from the [OMI realized library](https://oxford-man.ox.ac.uk/research/realized-library/) comprising of daily realized volatility computed from the intraday data of 31 stock indices where each index is treated as a time series. The volatility dataset is noisy with no definite seasonality and contains fewer observations than the previous datasets. It is used to contrast with the strongly seasonal electricity and traffic datasets.

* Content: Aggregated daily realized volatility computed from high-frequency intraday returns.
* Temporal Features: Exhibits weak seasonality and high noise.
* Key Challenges: Frequent abrupt regime changes and lack of regular patterns compared to energy or traffic data.

---
## SOLAR POWER

The solar power dataset is provided by [NREL8](https://www.nrel.gov/research/data-tools). The dataset exhibits daily seasonality and intermittent periods of zero power production during nighttime.

* Content: Solar power generation values.
* Temporal Features: Strong daily cycles driven by the sun, along with longer-term seasonal weather variations.
* Key Challenges: Values are strictly zero during nighttime; values during the day can vary greatly due to adverse weather.

---
## WIND POWER

The wind power dataset is collected from Kaggle, measuring the percentage wind power output of 29 european countries. The wind dataset is extremely noisy with slight yearly and monthly seasonality. Furthermore, in contrast to the other datasets, wind power is completely independent of known time inputs and will serve as an interesting comparison to the other datasets.

A dataset covering wind power generation across 29 countries, reflecting the stochastic nature of weather.

* Content: Normalized power output expressed as a percentage of total capacity.
* Temporal Features: High short-term variability with a much weaker seasonal structure than solar power.
* Key Challenges: Highly influenced by atmospheric conditions, resulting in frequent and unpredictable fluctuations.