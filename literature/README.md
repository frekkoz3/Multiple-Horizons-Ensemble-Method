# Literature review

This folder is dedicated to the exploration of already existing similar ideas in order to have a strong basis to begin the implementation of the novel method.

This folder contains several papers. In this section we will provide a brief summary of their contents.

## 📄 (about) The Combination of Forecasts

1983, *Robert L. Winkler  and  Spyros Makridakis*

> **Abstract** \
 Aggregating information by combining forecasts from two or more forecasting methods is an alternative to using just a single method. \
> In this paper we provide extensive empirical results showing that combined forecasts obtained through weighted averages can be quite accurate. \
> Five procedures for estimating weights are investigated, and two appear to be superior to the others. These two procedures provide forecasts that are more accurate overall than forecasts from individual methods. Furthermore, they are superior to forecasts found from a simple unweighted average of the same methods.

In this paper, the authors present extensive empirical evidence on the effectiveness of forecast combination through weighted averages. They evaluate five different procedures for estimating combination weights, originally proposed in the foundational paper *The Combination of Forecasts* of *Bates and Granger (1969)* and *Newbold and Granger (1974)*.

Assuming to have $M$ forecasts $f_{1, t}, ..., f_{M, t}$, the combined forecast is
$$
    \hat{y}_t = \sum_{i=1}^M{w_i f_{i_t}}\text{ ,  } \sum_{i=1}^M {w_i} = 1
$$

The five procedure to compute the weights proposed by *Bates and Granger (1969)* are:

- Simple (Equal) Weights, $w_i = \frac{1}{M}$
- Inverse Mean Squared Error (MSE) Weights, $w_i \propto \frac{1}{MSE_i}$
- Minimum Variance (Optimal) Weights, $w  = \frac{\Sigma^{-1}1}{1^T\Sigma^{-1}1}$, where $\Sigma$ is the covariance matrix of forecast errors
- Regression-Based Weights (Unconstrained OLS), estimate weights by regressing the target on forecasts $y_t = \beta_1 f_{1, t} + ... + \beta_M f_{M, t}$
- Constrained Regression (Sum-to-One), same as regression, but enforce $\sum_{i=1}^M {w_i} = 1$

The weighting methods are compared across multiple datasets and forecasting scenarios, and their performance is assessed relative to both individual forecasting models and a simple equal-weight average. In particular the 1001 time series used in the *Makridakis et al., (1982, 1983)* forecasting accuracy study were used.

The results show that two of the proposed weighting procedures consistently outperform the others, yielding combined forecasts that are more accurate than those produced by any single method as well as by unweighted combinations.

## 📄 Analyses of Global Monthly Precipitation using Gauge Observations

1996, *Xei*

> **Abstract**
Previous studies on multi-model ensemble forecasting mainly focused on the weight allocation of each model, but did not discuss how to suppress the reduction of ensemble forecasting accuracy when adding poorer models. Based on a variant weight (VW) method and the equal weight (EW) method, this study explored this topic through theoretical and real case analyses. A theoretical proof is made, showing that this VW method can improve the forecasting accuracy of
a multi-model ensemble, in the case of either the same models combination or adding an even worse model into the original multi-model ensemble, compared to the EW method. Comparative multi-model ensemble forecasting experiments against a real case between the VW and EW methods show that the forecasting accuracy of a multi-model ensemble applying the VW method is better than that of each individual model (including the model from the European Centre for Medium-Range
Weather Forecasts). The 2 m temperature forecasting applying the VW method is superior to that applying the EW method for all the multi-model ensembles. Both theoretical proof and numerical experiments show that an improved forecast, better than a best model, is generally possible.

**! TO COMPLETE! This under here is just a little introduction, an idea!**

## 📄 Multi-output Ensembles for Multi-step Forecasting

2023, *V Cerqueira, L Torgo*

> **Abstract** \
Ensemble methods combine predictions from multiple models to improve forecasting accuracy. \
> This paper investigates the effectiveness of multi-output ensembles for multi-step time series forecasting problems. \
> While dynamic ensembles have been extensively studied for one-step ahead forecasting, their application to multi-step forecasting remains largely unexplored, particularly regarding how combination rules should be applied across different forecasting horizons. \
> We conducted comprehensive experiments using 3568 time series from diverse domains and an ensemble of 30 multi-output models to address this research gap. \
> Our findings reveal that dynamic ensembles based on arbitrating and windowing techniques achieve the best performance according to average rank. \
> Interestingly, we observed that most dynamic approaches struggle to outperform a simple static ensemble that assigns equal weights to all constituent models, especially as the forecasting horizon increases. \
> **The performance advantage of dynamic methods is more pronounced in short-term forecasting scenarios**. The experiments are publicly available in a repository.

In this paper, the authors study dynamic ensembles for multi-step problems, and specifically explore how combination weights can vary with horizon. It considers strategies such as computing weights for each horizon independently and propagating them across the forecast horizon.

This work is pretty similar to our main goal: instead of one static weight per model, it acknowledges that goodness of models changes with forecasting horizon. It doesn’t use variance specifically, but it shows that computing different weights for each horizon is meaningful.

The authors used the combination of 40 different algorithms with differents parameters: random forest regression, extra trees regression, bagging of decision trees, projection pursuit regression, LASSO regression, ridge regression, elastic-net regression, k-nearest neighbors regression, principal components re gression, and partial least squares regression.

They tested 9 combination methods:

- **Simple**: Combination rule which assigns equal weights to all models
- **Window**: The weights are computed according to the forecasting performance in the last λ observations
- **Blast**: A variant of the Window approach. Instead of using past recent performance to weigh the available models, the idea is to select the model with the best performance in the last λ observations
- **ADE**: A dynamic combination approach based on a meta-learning strategy called arbitrating. The idea is to build a meta model (a Random Forest) for each (base) model in the ensemble. Each meta model is designed to predict the error of the corresponding base model. Then, the models in the ensemble are weighted  according to the error forecasts
- **EWA**: A dynamic combination rule based on an exponentially weighted average
- **FS**: The fixed share dynamic combination approach. This method is designed to handle non-stationary time series
- **MLpol**: A dynamic combination method based on a polynomially weighted average
- **Best**: A baseline which selects the individual model in the ensemble with the best performance in the training data to predict all the test instances
- **LossTrain**: Another baseline which weights the available models based on the error on the training set. The weights are static and fixed for all testing observations

In the paper the following approaches to estimate the weights at each time-step are proposed:

- **Complete Horizon** (CH): The weights of individual models are estimated using their average performance over the complete forecasting horizon
- **Individual Horizon** (IH): The ensemble estimates different weights for each horizon
- **First Horizon Forward** (FHF): The weights computed for the first horizon are propagated over the rest of the horizon
- **Last Horizon Backward** (LHB): According to LHB, the weights computed for the last horizon are propagated backward to all horizons before it.

All the tests were done on 3568 univariate time series.

Results show as it is not trivial to beat (on average) the simple combination method.

This study confirms that model performance varies across forecasting horizons and that horizon-aware weighting is meaningful. While most dynamic methods struggle to consistently outperform simple equal-weight ensembles, these findings motivate alternative approaches. Our work builds on this insight by proposing a variance-based horizon-dependent weighting strategy, focusing on predictive uncertainty rather than historical error alone.

## 📄 A Comparative Study of Multi-Model Ensemble Forecasting Accuracy between Equal- and Variant-Weight Techniques

2022, *Xiaomin Wei, Xiaogong Sun, Jilin Sun, Jinfang Yin, Jing Sun and Chongjian Liu*

> **Abstract**
Previous studies on multi-model ensemble forecasting mainly focused on the weight allocation of each model, but did not discuss how to suppress the reduction of ensemble forecasting accuracy when adding poorer models. \
> Based on a variant weight (VW) method and the equal weight (EW) method, this study explored this topic through theoretical and real case analyses. \
> A theoretical proof is made, showing that this VW method can improve the forecasting accuracy of a multi-model ensemble, in the case of either the same models combination or adding an even worse model into the original multi-model ensemble, compared to the EW method. \
> Comparative multi-model ensemble forecasting experiments against a real case between the VW and EW methods show that the forecasting accuracy of a multi-model ensemble applying the VW method is better than that of each individual model (including the model from the European Centre for Medium-Range Weather Forecasts). T\
> he 2 m temperature forecasting applying the VW method is superior to that applying the EW method for all the multi-model ensembles. Both theoretical proof and numerical experiments show that an improved forecast, better than a best model, is generally possible.

Even outside of horizon-specific work, there are empirical studies showing that Variant Weights (VW), based on variance of errors, can outperform equal-weight (EW) ensembles. \
For example, a comparative study in ensemble forecasting showed that weighting by a variant method (e.g., inverse-of-variance-of-errors based) leads to better average forecast accuracy than equal weights. \
A mathematical proof is given for demonstrate how VW ensemble methods provide, on average, better prediction than EW methods. The result is also supported by empirical experiments, applied to continuous temperature.

The number of models in the ensemble is pretty low (2-4). \
All the models used (taken from previous papers) are geophysical models, so they have nothing to do with ML or statistical ones.

## 📄 Performance metrics for multi-step forecasting measuring win-loss, seasonal variance and forecast stability: an empirical study

2024, *Eivind Strøm & Odd Erik Gundersen*

> **Abstract**
This paper addresses the evaluation of multi-step point forecasting models. Currently, deep learning models for multi-step forecasting are evaluated on datasets by selecting one error metric that is aggregated across the time series and the forecast horizon. This approach hides insights that would otherwise be useful for practitioners when evaluating and selecting forecasting models. We propose four novel metrics to provide additional insights when evaluating models: 1) a win-loss metric that shows how models perform across time series in the dataset , allowing the practitioner to check whether the model is superior for all series or just a subset of series. 2) a variance weighted metric that accounts for differences in variance across the seasonal period. It can be used to evaluate models for seasonal datasets such as rush hour traffic prediction, where it is desirable to select the model that performs best during the periods of high uncertainty. 3) a delta horizon metric measuring how much models update their forecast for a period in the future over the forecast horizon. Less change to the forecast means more stability over time and is desirable for most forecasting applications. 4) decomposed errors that relate the forecasting error to trend, seasonality, and noise. Decomposing the errors allows the practitioners to identify for which components the model is making more errors and adjust the model accordingly. To show the applicability of the proposed metrics, we implement four deep learning architectures and conduct experiments on five benchmark datasets. We highlight several use cases for the proposed metrics and discuss the applicability in light of the empirical results.

We include this paper because it provides a strong foundation for evaluating multi-step forecasting models and proposes novel metrics that address limitations of commonly used evaluation practices. In particular, the authors focus explicitly on metrics designed for multi-step forecasting, which aligns closely with the goals of our work.

Traditional forecasting evaluation relies on point-wise error metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and Mean Absolute Scaled Error (MASE). MSE and RMSE penalize large errors more heavily, while MAE provides a linear measure of average error magnitude. MAPE expresses errors as percentages but suffers from instability when true values are close to zero. MASE scales errors relative to a naïve forecasting method, making it comparable across different datasets and series.

In addition to absolute error metrics, relative performance against a naïve baseline forecast is widely used. Such baselines typically correspond to simple persistence or seasonal naïve models. Metrics like MASE and Root Mean Squared Scaled Error (RMSSE) explicitly quantify how much better a model performs compared to these baselines, allowing meaningful comparison across heterogeneous time series.

When evaluating multi-step forecasting models, performance is often summarized using aggregated metrics, obtained by averaging errors across both the forecasting horizon and all time series in the dataset. While this approach is convenient, it can obscure important information about how models behave across different horizons, series, or uncertainty regimes, reducing interpretability for practitioners.

To address these limitations, the authors propose several novel evaluation metrics aimed at providing more granular and informative insights:

- **Win–loss metric**: measures how often one model outperforms another across individual time series, rather than relying on average error. This allows practitioners to determine whether a model consistently performs better or only excels on a subset of series.
- **Variance-weighted error**: weights forecast errors according to the variance of the target series, placing greater emphasis on periods or series with higher variability. This is particularly useful for seasonal or heteroskedastic datasets, where accurate predictions during high-variance periods are more critical.
- **Delta horizon metric**: quantifies how much a model’s forecast for a given future time step changes as the forecast origin moves forward. Smaller changes indicate greater forecast stability, which is often desirable in operational settings.
- **Decomposed error metric**: decomposes the total forecasting error into components associated with trend, seasonality, and noise. This decomposition helps identify which structural aspects of the time series are most challenging for a given model.

The empirical results of the paper are not intended to rank forecasting models, but rather to demonstrate the significance and usefulness of the proposed metrics. The authors evaluate several models, including SARIMA, Seq2Seq networks, Temporal Convolutional Networks, DeepAR, and the Temporal Fusion Transformer, across five benchmark datasets: Electricity, Traffic, Volatility, Solar, and Wind.

The results show that different evaluation metrics can substantially alter conclusions about which model performs best. In particular, the proposed metrics often reveal differences in stability, robustness, and horizon-dependent behavior that are not captured by aggregated error measures. Overall, the novel metrics are shown to provide more reliable and informative assessments of multi-step forecasting performance, supporting their use alongside traditional error metrics.

## 📄 Time Series Forecasting With Neural Networks: A Survey

2020, *Bryan Lim and Stefan Zohren*

> **Abstract**
Numerous deep learning architectures have been developed to accommodate the diversity of time series datasets across different domains. In this article, we survey common encoder and decoder designs used in both one-step-ahead and multi-horizon time series forecasting – describing how temporal information is incorporated into predictions by each model. Next, we highlight recent developments in hybrid deep learning with neural network components to improve pure methods in either category. Lastly, we outline some ways in which deep learning can also facilitate decision support with time series data.

In this paper, the authors present a comprehensive survey of neural network–based approaches for time series forecasting. The survey covers a wide range of architectures, including recurrent, convolutional, attention-based, and hybrid encoder–decoder models, and discusses how temporal information is incorporated for both one-step-ahead and multi-horizon forecasting tasks.

An important aspect highlighted by the authors concerns multi-step forecasting strategies. They distinguish between two main approaches:

- **iterative (autoregressive) approach**, where a model predicts one step ahead and recursively feeds its own predictions to reach the desired forecasting horizon
- **direct approach**, where the model predicts the entire forecasting horizon in a single forward pass

The paper discusses the advantages and limitations of each strategy, particularly in terms of error accumulation and model complexity.

Overall, this survey provides a broad overview of neural network architectures and forecasting strategies that can be considered when designing multi-step forecasting methods, offering useful insights and guidance for the development of our approach. We will not summarize all the architectures here.

In particular, the discussion on iterative and direct forecasting strategies motivates our hybrid design, which combines horizon-specific training with an autoregressive generation mechanism.

## 📄 A Survey on Graph Neural Network for Time Series Forecasting, Classification, Imputation and Anomaly Detection

2024, *Ming Jin , Huan Yee Koh , Qingsong Wen , Daniele Zambon , Cesare Alippi, Geoffrey I. Webb ,Irwin King , Shirui Pan*

> **Abstract**
Time series are the primary data type used to record dynamic system measurements and generated in great volume by both physical sensors and online processes (virtual sensors). Time series analytics is therefore crucial to unlocking the wealth of information implicit in available data. With the recent advancements in graph neural networks (GNNs), there has been a surge in GNN-based approaches for time series analysis. These approaches can explicitly model inter-temporal and inter-variable relationships, which traditional and other deep neural network-based methods struggle to do. In this survey, we provide a comprehensive review of graph neural networks for time series analysis (GNN4TS), encompassing four fundamental dimensions: forecasting, classification, anomaly detection, and imputation. Our aim is to guide designers and practitioners to understand, build applications, and advance research of GNN4TS. At first, we provide a comprehensive task-oriented taxonomy of GNN4TS. Then, we present and discuss representative research works and introduce mainstream applications of GNN4TS. A comprehensive discussion of potential future research directions completes the survey. This survey, for the first time, brings together a vast array of knowledge on GNN-based time series research,  highlighting foundations, practical applications, and opportunities of graph neural networks for time series analysis.

We include this paper as it provides a comprehensive overview of how Graph Neural Networks (GNNs) can be applied to time series analysis. The survey illustrates how GNN-based models are able to capture inter-temporal and inter-variable dependencies, and discusses their application not only to forecasting tasks, but also to classification, imputation, and anomaly detection.

In particular, the survey highlights how different modeling assumptions can lead to varying uncertainty characteristics, which is relevant when considering ensemble-based forecasting strategies.

Although the paper is not directly related to our proposed method, it offers valuable background on advanced neural architectures for time series modeling and highlights alternative directions for incorporating structural information into forecasting models.
