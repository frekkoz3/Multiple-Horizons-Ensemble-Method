# Univariate Multiple Horizons Ensemble Method (U MHEMe)

This document is dedicated to the description of the univariate version of the Multiple Horizons Ensemble Method.

This document is divided in three parts:

1. The first part where we present the general idea of the method
2. The second part where we describe the practical implementation
3. The third part where we explore some subtle details about the implementation and about possible variants of the method

---

## General Idea

The Univariate Multiple Horizon Ensemble Method (U-MHEMe) is a time series forecasting technique designed to combine the advantages of direct forecasting, autoregressive modeling, and variance-based ensemble methods.

The method is motivated by the following observations:

- Direct forecasting methods can produce multi-step predictions up to a fixed horizon without recursive error propagation, resulting in horizon-dependent but non-compounding predictive variance.
- Autoregressive forecasting enables prediction over an unbounded horizon, at the cost of error accumulation and increasing predictive variance as the forecast horizon grows.
- Ensemble methods, when appropriately weighted and under the assumption of weakly correlated prediction errors, can reduce both prediction variance and forecast error.

The proposed approach integrates these three ideas by combining multiple direct forecasting models with different horizons through an autoregressive mechanism and aggregating their predictions using variance-based weighting.

## Formal Definition

Let \( X = \{x_0, \dots, x_t\} \) be a univariate time series. The goal is to predict the next \( h \) future values
\[
\hat{X}_f = \{\hat{x}_{t+1}, \dots, \hat{x}_{t+h}\}.
\]

### Direct Forecasting Models

Let \( M_k \) denote a **direct forecasting model** with horizon \( k \), defined as
\[
M_k : \mathbb{R}^w \rightarrow \mathbb{R}^k,
\]
where \( w \) is the input window size.

### Autoregressive Forecasting Operator

Define an autoregressive operator \( A_h(\cdot) \) that extends a direct forecasting model to a target horizon \( h \) by recursively feeding predictions back as inputs. Given a model \( M_k \), the operator produces a full \( h \)-step forecast:
\[
A_h(M_k)(X_{t-w+1:t}) = \hat{X}^{(k)}_f \in \mathbb{R}^h.
\]

The autoregressive mechanism iteratively applies \( M_k \) until \( h \) predictions are obtained.

### Model Ensemble Construction

Instantiate a family of direct models with increasing horizons:
\[
\mathbb{M}_h = \{ M_1, M_2, \dots, M_h \}.
\]

Applying the autoregressive operator to each model yields:
\[
\mathbb{F}_h = \{ A_h(M_1), A_h(M_2), \dots, A_h(M_h) \}.
\]

Each \( A_h(M_i) \) produces a full forecast vector
\[
\hat{X}^{(i)}_f = \left( \hat{x}^{(i)}_{t+1}, \dots, \hat{x}^{(i)}_{t+h} \right).
\]

### Prediction Variance Estimation

For each model \( i \in \{1,\dots,h\} \) and forecast step \( f \in \{1,\dots,h\} \), estimate the prediction variance
\[
v_{i,f} = \mathrm{Var}(\hat{x}^{(i)}_{t+f}),
\]
using an appropriate uncertainty estimation method (e.g. jackknife resampling, bootstrap, or meta-learning).

### Variance-Weighted Aggregation

Assuming uncorrelated prediction errors across models, the final forecast at horizon step \( f \) is obtained via inverse-variance weighting:
\[
\hat{x}_{t+f} =
\frac{
\sum_{i=1}^{h} \frac{1}{v_{i,f}} \, \hat{x}^{(i)}_{t+f}
}{
\sum_{i=1}^{h} \frac{1}{v_{i,f}}
}.
\]

This produces the final prediction vector
\[
\hat{X}_f = \{\hat{x}_{t+1}, \dots, \hat{x}_{t+h}\}.
\]

---

## Implementation

In this section we discuss about the actual implementation.\
We present:

- The direct model we decided to use
- How we compute the weights (both how to compute the variance and what data are used in order to compute them)
- What benchmarks have beeen selected to evaluate the model

### Direct model

### Weights computation

### Benchmarks

---

## Details

---
