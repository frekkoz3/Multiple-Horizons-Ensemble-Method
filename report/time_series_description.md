# An introduction to Time Series

The scope of this file is to introduce the reader to the main characteristics of Time Series and their Analysis. \
Most of the information included derives from the first chapters of "*Time Series Analysis and Its Applications*", *R. H. Shumway, D. S. Stoffer, Springer, 2016*. \
In Section 1, an introduction to some basic concepts is presented; in Section 2, some classic statistical models for Time Series are described.

---

## 1. Basic Concepts
A *Time Series* usually consists on a collection of adjacent points ${x_0, x_1, ...}$, indexed accordingly to the order they are obtained over time (${t_0, t_1, ...}$). \
Usually, such points are treated as *random variables*, ${x_t}$, observed realizations of a *stochastic process*. \
Many of time series are actually *continuous time series*, that are discretized and approximated by the observations (*discrete time parameter series*); an insufficient sampling rate leads to a critic phenomenon called *aliasing*.

The act of answering mathematical and statistical problems about a given Time Series is called *Time Series Analysis*. \
Two approaches for doing Time Series Analysis, different but not mutually exclusive, exist: the *time domain approach* investigates lagged relationships between data, while *frequency domain approach* deals with cycles or periods among data. 

Analysis techniques rely on some assumptions. One of them is that continuous time series can be specified in terms of some finite-dimensional distribution functions, allowing to ... [TODO].

### 1.1 Measures of Dependence
A classic assumption in statistical inference is that data sampled are *independent and identically distributed* (*iid*); in time series analysis such assumption is clearly impossible to hold, since data collected over time usually share some properties, thus leading to *dependence*. \
There exists a variety of measures of dependence:
- The **mean function**, defined as
$$
    \mu_{x_t} = \mathbb{E}[x_t] = \int_{-\infty}^{+ \infty} x f_t(x) dx
$$
- The **autocovariance function**, defined as 
$$
    \gamma_X(s, t) = \mathbb{C}(x_s, x_t) = \mathbb{E}[(x_s - \mu_s)(x_t - \mu_t)],    
$$
provides a measure of *linear* dependence among two points $x_s$ and $x_t$ of the same time series $X$; smooth series exhibit high autocovariance values even for far $t, s$, while choppier series tend to have autocovariance values nearly zero for most of the $(s, t)$ pairs. Autocovariance reduces to *variance* in the case of $s = t$.
- The **autocorrelation function**, defined as
$$ \rho(s, t) = \frac{\gamma (s, t)}{\sqrt{\gamma (s, s) \gamma (t, t)}},$$
is a measure for the linear predictability of the series at time t, $x_t$, using only the value $x_s$.

Some dependence can also be shown by two or more different time series. Think, e.g., to some measures of soil temperature and soil humidity: clearly, if the temperature is high, we expect the soil to by drier, i.e., showing low humidity. \
In such cases can be used two slight modifications of the measures yet introduced:
- The **cross-covariance function**, between two series $X$ and $Y$, is defined as 
$$\gamma_{XY}(s, t) = \mathbb{C}[x_s, y_t] = \mathbb{E}[(x_s - \mu_{Xs})(y_t - \mu_{Yt})]$$
- The **cross-correlation function** is instead given by
$$\rho_{X, Y}(s, t) = \frac{\gamma_{XY}(s, t)}{\sqrt{\gamma_X (s, s) \gamma_Y (t, t)}}$$
Extension are allowed also for more than two series (*multivariate time series*).

### 1.2 Stationarity
For allowing some kind of forecasting over a collection of sampled data, we can assume that the time series hides a sort of regularity over time. \
A key concept in such context is the **stationarity**. In practice, a time series is *stationary* if its properties (mean, variance, covariance, ...) do not change over time, e.g. $\mathbb{E}[x_t] = \mu \ \forall t$.

A more formal definition is here provided.

The **strict stationarity** is a property of a time series $X$ for which the joint distribution over any collection of values $x_{t_1}, x_{t_2}, ..., x_{t_k}$ is identical to that of the values collected at a time shifted by some $h$ constant, $x_{t_1 + h}, x_{t_2 + h}, ..., x_{t_k + h}$. \
The property implies that $\mu_t$ is constant for all $t$, and that $\gamma (s, t) = \gamma (s+h, t+h)$. 

The **weak stationarity** refers instead to time series with finite variance such that:
1. $\mu_t$ does not depend on time $t$, hence $\mu_t = \mu \forall t$
2. $\gamma (s_t)$ depends only on the difference $h = |s - t|$, hence $\gamma (t+h, t) = \gamma (h, 0)$.
If a time series is weakly stationary, (or, in the following, simply stationary), its mean and autocorrelation functions can be estimated by averaging.

The concept of (weak) stationarity forms the basis of many analysis procedures.

### 1.3 Time Series Decomposition
Time Series usually show **trends** $\mathbf{T}$, i.e., they tend to increase or decrease over time, and/or **seasonalities** $\mathbf{S}$, i.e., they tend to have a repeated pattern over time.
These two elements are the so-called *deterministic components* of a Time Series and can be extracted through a *decomposition* process.
What remains after such a decomposition is a *random component*, $\mathbf{R}$, non-deterministic and so usually referred to as "white noise". \


>⚠️️️ ️Note that if a Time Series shows a trend or a seasonality, it means that the mean and the variance are shifting over time: the Time Series is **non-stationary**!⚠️️️

There exists a variety of methods for Time Series Decomposition (or so-called *Smoothing Methods*):
1. **Center Moving Average Method**: works for estimating the trend. Said $p$ the period of the seasonality, the trend is derived as:
$$
    \mathbf{T}_t = (x_{t-d} + x_{t-d+1} + ... + x_{t-d+d}) / p,
$$
where $d = (p - (p \mod 2 )) / 2$
2. **Holt-Winters Smoothing**: estimates three parts: the *level* $\mathbf{L}$, the *slope* $\mathbf{B}$ and the *seasonality* of the Time Series. Note how level and slope are better specifications for a linear trend. There exist many approaches; one of the most common is based on some smoothing parameters $\alpha, \beta, \theta, p$:
$$
\begin{align}
    \mathbf{L}_t &= \alpha(x_t - \mathbf{S}_{t-p}) + (1 - \alpha)(\mathbf{L}_{t-1} + \mathbf{B}_{t-1}) \\
    \mathbf{B}_t &= \beta(\mathbf{L}_t - \mathbf{L}_{t-1}) + (1 - \beta)\mathbf{B}_{t-1} \\
    \mathbf{S}_t &= \gamma (x_t - \mathbf{L}_t) + (1-\gamma)\mathbf{S}_{t-p}
\end{align}
$$
3. Modern approaches include kernel smoothing, polynomial fitting, smoothing splines, ... .

---

## 2. Classic Time Series models
The following section discusses peculiarities of the most-widely used classic models used for describing and analyzing time series. \
Subsection 2.1 introduces very basilar models, while Subsection 2.2 describes some more complex statistical structures. Subsection 2.3 presents more recent models, developed in the machine learning field.

### 2.1 Simple time series generators
Here some common generators of time series samples are presented. \
Despite being extremely naïve and useless in practice, they are the base of much more complex structures that allow to capture real-world phenomena. \
Simple properties of these models are presented but not proved.

The parameters of the following models can be estimated by simply applying classical Least Squares Errors, or using Maximum Likelihood Estimate under the assumption of gaussianity of the time series.


#### 2.1.1 White Noise
*White Noise* is without any doubt the simples kind of time series. \
It consists of a sequence of uncorrelated random variables, $x_t$, coming from a white-noise process $\text{wh}_X$ with zero mean and constant finite variance $\sigma^2_X$. \
Usually, such variables are assumed to be normally distributed, i.e., $x_t \sim \mathcal{N}(0, \sigma^2)$; in such a case, we properly talk about *Gaussian White Noise*.

The autocorrelation matrix $\Gamma$ over a time series $X$ is here such that $\Gamma = diag (\sigma^2_X, \sigma^2_X, ..., \sigma^2_X)$. 


#### 2.1.2 Moving Average (MA) Models
A simple extension of White Noise time series is given by replacing the $x_t$ with its **moving average**, an average that takes care of the value itself, its past value(s) and its future value(s), with some fixed *order* (like a "window") $q$. \
For the case of $q = 1$, one gets $x_t = (w_{t-1}, w_t, w_{t+1}) / 3$. 

Values $w_t$ are always assumed to be drawn from a white noise process $\text{wh}_W (0, \sigma^2_W)$. \

MA models can be improved by applying some weights $\theta_t$ to get more expressive functions that the ones allowed by pure averaging.
Given an order $q$, one gets:
$$
    x_t = \mu + w_t + \theta_1 w_{t-1} + \theta_2 w_{t-2} + ... + \theta_q w_{t-q}
$$

The following properties can be proven:
1. $\mathbb{E}[x_t] = \mu$, so MA models are stationary
2. $\gamma_X(t, t) = \mathbb{V}[x_t] = \sigma_W^2 (1 + \theta^2_1 + ... + \theta^2_q)$
3. $\gamma_X(t, t+k) = \sigma^2_W \sum_{i=0}^{q-k} \theta_i \theta_{i+k}$ if $0 \leq k \leq q$, else $0$.

For practical reasons, MA models are required to be *invertible*, i.e., one can extract a noise value $w_t$ as a (finite) linear combination of samples ${x_t, x_{t-1}, ..., x_0}$.


#### 2.1.3 Autoregressive (AR) Models
An **autoregressive model** is a time series in which the current value $x_t$ depends linearly on its previous values, plus a stochastic term $w_t$ coming from a white noise process $\text{wh}_W(0, \sigma^2_W)$. \
For an order of $p = 2$, 
$$
    x_t = x_{t-1} + x_{t-2} + w_t
$$

Also in this case some weights $\psi_t$ can be introduced, so that:
$$
    x_t = \psi_0 + \psi_1 x_{t-1} + \psi_2 x_{t-2} + ... + \psi_p x_{t-p} + w_t
$$

Some properties:
1. $\mathbb{E}[x_s w_t] = 0$
2. $\mathbb{E}[x_t | x_{(t-p):(t-1)}] = \psi_0 \sum_{i=1}^p \psi_i x_{t-i}$
3. $\mathbb{V}[x_t | x_{(t-p):(t-1)}] = \mathbb{V}[w_t] = \sigma_W^2$

Furthermore, if AR model is also stationary, it holds:
1. $\mathbb{E}[x_t] = \psi_0 / (1 - \psi_1 - ... - \psi_p)$
2. $\mathbb{E}[x_t w_t] = \sigma^2_W$

If the sum of an AR model's weights is finite, then the model is called *causal*.


### 2.2 Classical Methods
The following subsection presents improved classical models that combine in different (but similar) ways the concepts previously introduced.
A brief description of the models' properties is provided, but not proved, as well.

#### 2.2.1 Autoregressive Moving Average (ARMA) Models
The idea of ARMA models is to combine AR and MA properties and definitions:
$$
    x_t = \psi_0 + \psi_1 x_{t-1} + ... + \psi_p x_{t-p} + w_t + \theta_1 w_{t-1} + ... + \theta_q w_{t-q},
$$

where, as usual, we are defining an ARMA models of orders (p, q) and $w_t \sim \text{wh}(0, \sigma^2_W)$

Properties:
1. ARMA(p, q) is stationary if and only if ARMA(p, 0) is stationary.
2. ARMA(p, q) is causal if and only if ARMA(p, 0) is causal.
3. ARMA(p, q) is invertible if and only if ARMA(0, q) is invertible.
4. (Non formally), ARMA models have closed solutions for $\mathbb{E}[\cdot], \gamma, \rho$.

> ⚠️ For building a model given the samples, these **should be stationary** (use some checks, e.g. `statmodels.tsa.stattools.adfuller` method). If not, use models like the following. ⚠️


#### 2.2.2 AR - Integrated - MA (ARIMA) Models
ARIMA models have the same flavor of ARMA ones and become relevant once a time series shows some kind of trend. 

The idea of ARIMA models is to remove the trend from the samples to make the model stationary. Than, all the techniques applicable to ARMA are allowed. \
To capture trend, ARIMA looks at short-term (linear) correlations. \
In practice ARIMA models simply differentiate across consecutive points up to some degree $d$. This allows removing a polynomial trend of degree $d$ from the data. \

ARIMA models are specified by three orders $(p, d, q)$.

To give an example, if I have a time series $X$ with samples ${x_0, x_1, ..., x_t}$, than what ARIMA(p, d, q) does is simply build *another* time series $Y$ where each element is defined as:
$$
    y_t = (x_t - x_{t-1}) - (x_{t-1} - x_{t-2}) - ... - (x_{t-d+1} - x_{t-d})
$$.

Once derived the elements, the new time series $Y$ becomes an ARMA(p, q).


#### 2.2.3 Seasonal - ARIMA (SARIMA) Models
SARIMA models allow the tractability of time series showing some kind of seasonality.
The idea of SARIMA is someway similar to the ARIMA one: we want to explicitate some relationship between data "at the same period of the seasonality", e.g., linking temperature of Jan '26 to Jan '25.

SARIMA models are specified by parameters $(p, d, q) \times (P, D, Q)_S$, where the first triplet $(p, d, q)$ refers to non-seasonal components, while the last triplet $(P, D, Q)_S$, instead, allows capturing long-term and periodical behaviors of range $S$.

In practice, once again what SARIMA models do is simply differentiate both through short- and long- term samples; given a time series $X$, SARIMA gets a new time series $Y$ where each element is defined as:
$$
    y_t = (x_t - x_{t-1}) - ... - (x_{t-d+1} - x_{t-d}) - (x_{t-S} - x_{t-S-1}) - ... - (x_{t-S-D+1} - x_{t-S-D})
$$

Both ARIMA and SARIMA can be expanded with the introduction of exogenous ("external") variables.
Finally, ARIMA and SARIMA are part of a family of models called **Box-Jenkins** models.


#### 2.2.4 AutoRegressive Conditional Heteroscedasticity (ARCH) Models
A completely different approach to time series consists in focusing on their time-varying variance, or **volatility**.
An ARCH model is defined by a couple of euqations and by an order $p$:
$$
\begin{cases}
    x_t = \sigma_t w_t \\
    \sigma_t^2 = \omega + \sum_{i=1}^p \alpha_i X_{t-i}^2
\end{cases},
$$

where $\omega, \alpha_i \geq 0$, $w_t \sim \text{wh}(0, 1) iid$ and $\sum_i \alpha_i < 1$.
The first equation explicitates the *mean*, the second the *variance* of the process.

It can be proved that $\sigma^2_t$ evolves accordingly to the previous values of $x_{t' < t}$, like "collecting" some information from the past for reflecting some variance in the present (and in the future). \
However, ARCH models also take care of the *unconditional variance*, supposed to be fixed to some $\bar{\sigma}^2$. The unconditional variance is defined as $\mathbb{E}[\sigma^2_{t+k} | {x_0, ..., x_t}] \xrightarrow{k \rightarrow \infty} \bar{\sigma}^2$. \
In practice, it means that for long-term analysis fluctuations are almost completely due by the unconditional variance, while for the short term a more proper "local" measure is preferable. \
This complex behavior is the reason why ARCH models are suited for dealing for financial time series.

A Generalized version of ARCH models (GARCH), also links $\sigma_t^2$ to its (weighted) previous values $\sigma_{t-j}^2$.



