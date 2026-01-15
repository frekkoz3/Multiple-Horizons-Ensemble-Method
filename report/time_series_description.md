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
    \mu_{x_t} = \mathbf{E}[x_t] = \int_{-\infty}^{+ \infty} x f_t(x) dx
$$
- The **autocovariance function**, defined as 
$$
    \gamma_X(s, t) = \mathbf{C}(x_s, x_t) = \mathbf{E}[(x_s - \mu_s)(x_t - \mu_t)],    
$$
provides a measure of *linear* dependence among two points $x_s$ and $x_t$ of the same time series $X$; smooth series exhibit high autocovariance values even for far $t, s$, while choppier series tend to have autocovariance values nearly zero for most of the $(s, t)$ pairs. Autocovariance reduces to *variance* in the case of $s = t$.
- The **autocorrelation function**, defined as
$$ \rho(s, t) = \frac{\gamma (s, t)}{\sqrt{\gamma (s, s) \gamma (t, t)}},$$
is a measure for the linear predictability of the series at time t, $x_t$, using only the value $x_s$.

Some dependence can also be shown by two or more different time series. Think, e.g., to some measures of soil temperature and soil humidity: clearly, if the temperature is high, we expect the soil to by drier, i.e., showing low humidity. \
In such cases can be used two slight modifications of the measures yet introduced:
- The **cross-covariance function**, between two series $X$ and $Y$, is defined as 
$$\gamma_{XY}(s, t) = \mathbf{C}[x_s, y_t] = \mathbf{E}[(x_s - \mu_{Xs})(y_t - \mu_{Yt})]$$
- The **cross-correlation function** is instead given by
$$\rho_{X, Y}(s, t) = \frac{\gamma_{XY}(s, t)}{\sqrt{\gamma_X (s, s) \gamma_Y (t, t)}}$$
Extension are allowed also for more than two series (*multivariate time series*).

### 1.2 Stationarity
For allowing some kind of forecasting over a collection of sampled data, we can assume that the time series hides a sort of regularity over time. \
A key concept in such context is the **stationarity**. In practice, a time series is *stationary* if its properties (mean, variance, covariance, ...) do not change over time. A more formal definition is here provided.

The **strict stationarity** is a property of a time series $X$ for which the probabilistic behaviour of every collection values $x_{t_1}, x_{t_2}, ..., x_{t_k}$ is identical to that of the time shifted by some $h$ constant, $x_{t_1 + h}, x_{t_2 + h}, ..., x_{t_k + h}$. \
The property implies that $\mu_t$ is constant for all $t$, and that $\gamma (s, t) = \gamma (s+h, t+h)$. 

The **weakl stationarity** refers instead to time series with finite variance such that:
1. $\mu_t$ does not depend on time $t$, hence $\mu_t = \mu \forall t$
2. $\gamma (s_t)$ depends only on the difference $h = |s - t|$, hence $\gamma (t+h, t) = \gamma (h, 0)$.
If a time series is weakly stationary, (or, in the following, simply stationary), its mean and autocorrelation functions can be estimated by averaging.

The concept of (weak) stationarity forms the basis of many analysis procedures.

---

## 2. Classic Time Series models
The following section discusses peculiarities of the most-widely used classic models used for describing and analyzing time series. \
Subsection 2.1 introduces very basilar models, while Subsection 2.2 describes some more complex statistical structures. Subsection 2.3 presents more recent models, developed in the machine learning field.

### 2.3 Simple kinds of time series
Here some common generators of time series samples are presented. \
Despite being extremely naive and useless in practice, they are the base of much more complex structures that allow to capture real-world phenomena.

#### 2.3.1 White Noise
*White Noise* is without any doubt the simples kind of time series. \
It consists of a sequence of uncorrelated random variables, $x_t$, coming from a white-noise process $\text{wh}_X$ with zero mean and constant finite variance $\sigma^2_X$. \
Usually, such variables are assumed to be normally distributed, i.e., $x_t \sim \mathcal{N}(0, \sigma^2)$; in such a case, we properly talk about *Gaussian White Noise*.

The autocorrelation matrix $\Gamma$ over a time series $X$ is here such that $\Gamma = diag (\sigma^2_X, \sigma^2_X, ..., \sigma^2_X)$. 

#### 2.3.2 Moving Averages
A simple extension of White Noise time series is given by replacing the $x_t$ with its **moving average**, an average that takes care of the value itself, its past value(s) and its future value(s), with some fixed "window" $w$. \
For the case of $w = 1$, one gets $v_t = (x_{t-1}, x_t, x_{t+1}) / 3$. 

Values $x_t$ are always assumed to be drawn from a white noise process $\text{wh}_X (0, \sigma^2_X)$. \
It holds that $$\mu_{v_t} = \mathbf{E}[v_t] = 1/3 [\mathbf{E}[x_{t-1}] + \mathbf{E}[x_{t}] + \mathbf{E}[x_{t+1}]] = 0.$$
Also the autocovariance matrix has fixed values. In particular, one can (simply) prove that:
$$
\gamma_X(s, t) =
\begin{cases}
\frac{1}{3}\sigma_X^2 & \text{if } |s - t| = 0,\\
\frac{2}{9}\sigma_X^2 & \text{if } |s - t| = 1,\\
\frac{1}{9}\sigma^2_X & \text{if } |s - t| = 2,\\
0 & \text{if } |s - t| > 2.
\end{cases}
$$

This approach leads to much smoother plots than white noise.

#### 2.3.3 Autoregressions
An **autoregression** is a time series model in which the current value $x_t$ depends linearly on its previous values, plus a stochastic term $s_t$ coming from a white noise process $\text{wh}_S(0, \sigma^2_S)$. \
For a window of $w = 2$, $$x_t = x_{t-1} + x_{t-2} + s_t$$.

A possible improvement of an autoregressive model is given by the introduction of a *drift*, that leads to the so called *random-walks models with drift*.
The drift is simply a fixed value $\delta \neq 0$, that "shifts" data upper or lower:
$$x_t = \delta + x_{t-1} + x_{t-2} + s_t$$.

It holds that $$\mu_{x_t} = \mathbf{E}[x_t] = \delta t + \sum_{j=1}^t \mathbf{E}[s_j] = \delta t,$$
and
$$\gamma_X(p, q) = \mathbf{C}[\sum_{j=1}^p s_j, \sum_{k=1}^q s_k] = \min_{p, q} \sigma_S^2$$

Autoregressive models are widely used in practice, since they can capture trends in data.



