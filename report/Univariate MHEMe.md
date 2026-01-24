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

Let $ X = \{x_0, \dots, x_t\} $ be a univariate time series. The goal is to predict the next $ h $ future values
$$
\hat{X}_f = \{\hat{x}_{t+1}, \dots, \hat{x}_{t+h}\}.
$$

### Direct Forecasting Models

Let $ M_k $ denote a **direct forecasting model** with horizon $ k $ , defined as
$$
M_k : \mathbb{R}^w \rightarrow \mathbb{R}^k,
$$
where $ w $ is the input window size.

### Autoregressive Forecasting Operator

Define an autoregressive operator $ A_h(\cdot) $ that extends a direct forecasting model to a target horizon $ h $ by recursively feeding predictions back as inputs. Given a model $ M_k $, the operator produces a full $ h $-step forecast:
$$
A_h(M_k)(X_{t-w+1:t}) = \hat{X}^{(k)}_f \in \mathbb{R}^h.
$$

The autoregressive mechanism iteratively applies $ M_k $ until $ h $ predictions are obtained.

### Model Ensemble Construction

Instantiate a family of direct models with increasing horizons:
$$
\mathbb{M}_h = \{ M_1, M_2, \dots, M_h \}.
$$

Applying the autoregressive operator to each model yields:
$$
\mathbb{F}_h = \{ A_h(M_1), A_h(M_2), \dots, A_h(M_h) \}.
$$

Each $ A_h(M_i) $ produces a full forecast vector
$$
\hat{X}^{(i)}_f = \left( \hat{x}^{(i)}_{t+1}, \dots, \hat{x}^{(i)}_{t+h} \right).
$$

### Prediction Variance Estimation

For each model $ i \in \{1,\dots,h\} $ and forecast step $ f \in \{1,\dots,h\} $, estimate the prediction variance
$$
v_{i,f} = \mathrm{Var}(\hat{x}^{(i)}_{t+f}),
$$
using an appropriate uncertainty estimation method (e.g. jackknife resampling, bootstrap, or meta-learning).

### Variance-Weighted Aggregation

Assuming uncorrelated prediction errors across models, the final forecast at horizon step $ f $ is obtained via inverse-variance weighting:
$$
\hat{x}_{t+f} =
\frac{
\sum_{i=1}^{h} \frac{1}{v_{i,f}} \, \hat{x}^{(i)}_{t+f}
}{
\sum_{i=1}^{h} \frac{1}{v_{i,f}}
}.
$$

This produces the final prediction vector
$$
\hat{X}_f = \{\hat{x}_{t+1}, \dots, \hat{x}_{t+h}\}.
$$

---

## Implementation

In this section we discuss about the actual implementation.\
We present:

- The direct model we decided to use
- How we compute the weights (both how to compute the variance and what data are used in order to compute them)
- What benchmarks have beeen selected to evaluate the model

### Direct model

The model selected as the base learner for the proposed MHEMe framework is the **Temporal Convolutional Network (TCN)**, with specific design choices motivated by the multi-horizon forecasting setting.

In particular, a **Horizon-Aware Huber Loss** is employed to explicitly control the contribution of prediction errors at different forecast horizons, enabling the estimation of horizon-dependent predictive variances.

Additionally, benchmark-specific preprocessing techniques and hyperparameters-tuning are applied in order to improve the performance of individual models. A detailed description of these preprocessing steps is provided in the [Benchmarks](#benchmarks) section.

#### Temporal Convolutional Network (TCN)

A Temporal Convolutional Network (TCN) is a convolutional architecture designed for sequence modeling and time series forecasting. Unlike recurrent models, TCNs rely exclusively on one-dimensional convolutional layers while preserving the temporal ordering of the input sequence.

Formally, given an input sequence $ x \in \mathbb{R}^{T} $, a TCN applies **causal convolutions**, ensuring that the output at time step $ t $ depends only on inputs $ \{x_0, \dots, x_t\} $. A causal convolution with kernel $ k $ and dilation factor $ d $ is defined as
$$
y_t = \sum_{i=0}^{K-1} k_i \, x_{t - d \cdot i}.
$$

To efficiently model long-range temporal dependencies, TCNs employ **dilated convolutions**, where the dilation factor increases exponentially across layers. This allows the receptive field to grow exponentially with network depth while maintaining a fixed computational cost.

The architecture is typically composed of stacked convolutional blocks with residual connections, enabling stable training and effective gradient propagation. Due to their parallelizable structure and stable gradients, TCNs are well-suited for direct multi-step forecasting tasks and serve as a robust base model for the proposed ensemble method.

#### Horizon-Aware Huber Loss

To account for the increasing uncertainty associated with longer forecast horizons, we adopt a **Horizon-Aware Huber Loss**, which extends the standard Huber loss by introducing horizon-dependent weighting.

Given the prediction error at horizon step $ f $,
$$
e_f = \hat{x}_{t+f} - x_{t+f},
$$
the standard Huber loss is defined as
$$
\mathcal{L}_{\delta}(e_f) =
\begin{cases}
\frac{1}{2} e_f^2, & \text{if } |e_f| \leq \delta \\
\delta \left( |e_f| - \frac{1}{2} \delta \right), & \text{otherwise}.
\end{cases}
$$

In the horizon-aware formulation, each horizon step is weighted by a non-negative coefficient $ w_f $, yielding the final loss:
$$
\mathcal{L} = \sum_{f=1}^{h} w_f \, \mathcal{L}_{\delta}(e_f).
$$

This formulation allows the model to explicitly emphasize or de-emphasize errors at specific horizons, enabling better control over the bias–variance trade-off across the forecast horizon. As a consequence, the resulting models exhibit horizon-dependent error distributions, which are later exploited by the MHEMe framework through variance-based aggregation.

### Weights computation

The computation of aggregation weights is a fundamental component of the proposed MHEMe framework, as it directly determines how predictions from different models are combined at each forecast horizon.

In its most straightforward implementation, the weight associated with model \( i \) at forecast step \( f \) is defined as the inverse of the empirical variance of its prediction errors, estimated on a validation set:
$$
w_{i,f} = \frac{1}{v_{i,f} + \varepsilon},
\qquad
v_{i,f} = \mathrm{Var}\left( \hat{x}^{(i)}_{t+f} - x_{t+f} \right),
$$
where $ \varepsilon > 0 $ is a small constant introduced for numerical stability.

The empirical error variance estimation represents a simple and interpretable approach for computing the aggregation weights. When a sufficiently large amount of data is available, this method provides stable and reliable variance estimates. However, it implicitly assumes stationarity of the error distribution and may become sensitive to limited sample sizes, potentially leading to noisy or biased variance estimates in low-data regimes.

In addition, empirical evidence suggests that estimating the weights exclusively on the validation set may lead to suboptimal generalization performance. In particular, computing the error variances on the training set often yields more stable estimates, resulting in improved performance on unseen test data.

To balance robustness and generalization, we additionally consider estimating the weights on a mixed set obtained by randomly sampling from both the training and validation datasets. This strategy reduces the variance of the estimated weights while mitigating potential overfitting to the training data.

The resulting horizon-dependent weights are then used to aggregate model predictions according to the variance-weighted ensemble rule defined in the [Formal Definition](#formal-definition) section.

Alternative strategies could be employed to estimate the horizon-dependent variances used for weight computation. Non-parametric approaches such as bootstrap or jackknife resampling may be used to directly estimate predictive uncertainty by repeatedly retraining or evaluating the base models on resampled datasets.

In particular, bootstrap-based variance estimation could provide a more expressive characterization of model uncertainty, especially in the presence of non-Gaussian error distributions. However, in time series settings, standard bootstrap techniques must be adapted (e.g., block bootstrap) to preserve temporal dependencies, significantly increasing computational cost.

Given the large number of models and forecast horizons involved in the MHEMe framework, we adopt empirical error variance estimation as a computationally efficient and sufficiently accurate approximation. Exploring more advanced variance estimation techniques is left as future work.

### Benchmarks

---

## Details

---
