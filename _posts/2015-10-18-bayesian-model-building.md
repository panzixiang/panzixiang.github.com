---
layout: post
title: "Bayesian model building"
description: ""
category: 
tags: []
---
{% include JB/setup %}
## Improving Heuristics for GDELT Feature Engineering

From the previous week, our pre-processing generated numerical data that describes the positive and the negative impact from different actors in the following format:

|Date                | USA_BUS_positive  | USA_BUS_negative | GBR_BUS_positive  | GBR_BUS_negative | ... |
|:---------------------|:-----------------|:----------------|:-----------------|:----------------|:----------|
|20050101    | +10.5              |-24.6         | +17.5              |-8.6   | ...      |    
|20050102    | +25.2              |-16.7         | +15.3              |-9.9  | ... |            
|...                   |                 |                 |        |                 |     |

We had three heuristics that can be used to measure the impact which are listed below. The problem with using the number of articles is that it is very sensitive to noise. The total number of news articles also varies from day to day.

#### First heuristic

$$ H_1 = NumArticles \times GoldSteinScale $$

#### Second heuristic

$$ H_2 = NumArticles \times AvgTone $$

#### Third heuristic

$$ H_3 = NumArticles \times QuadClass $$

One possible approach is to normalize the number of articles by the total number of articles in the same day. We tried fitting the number of articles within each day to a Poisson distribution and to a Pareto distribution:

<center><img src="/assets/week_4/num_articles_pareto.png" width="100%"></center>


<center><img src="/assets/week_4/num_articles_poisson.png" width="100%"></center>

We can then map the number of articles to a percentile using the CDF. Then each percentile can be mapped to a convex function (e.g. quadratic function)


## Gaussian Process Regression


We performed Gaussian Process Regression on the historical exchange rates between GBP and USD in 2005. In the first image below, we used the first 40 days of 2005 as the training data to predict the next 10 days' exchange rates.


<center><img src="/assets/week_4/gaussian_process_40_days.png" width="100%"></center>

In the second image, we used the first 50 days of 2005 as the training data to predict the next 10 days' exchange rates.

<center><img src="/assets/week_4/gaussian_process_50_days.png" width="100%"></center>

In the second image, we used the first 70 days of 2005 as the training data to predict the next 10 days' exchange rates.

<center><img src="/assets/week_4/gaussian_process_70_days.png" width="100%"></center>


### Binary classification with Gaussian Process

We use Gaussian Process Regression to frame this as a binary classification problem: use the historical prices to predict the mean of the next day's price:

$$ prediction(T+1) = E[F_{T+1}| F_1, ... , F_T] $$

If the prediction is more than the current price, predict an upward trend; otherwise, predict a downward trend.

The table below shows the classification accuracy of running kernel SVM with the different heuristics on the test set:

|Number of training days         | Best correlation function  | Test accuracy |
|:------------------------:|:-----------------|:---------------:|
| 60 | Linear   | 47.0%   |    
| 30 | Absolute exponential   | 48.0%   |
| 10 | Absolute exponential  | 48.3%   |
| 3 | Squared exponential  | 49.7%   |

The next step: to look into how to augment Gaussian Process regression with exogenous inputs. See the paper [Augmented Functional Time Series Representation and Forecasting with Gaussian Processes](http://papers.nips.cc/paper/3324-augmented-functional-time-series-representation-and-forecasting-with-gaussian-processes.pdf) by Yoshua Bengio

Examples of exogenous inputs:

* GDELT news data

* Interest rates

* Inflation rates

## ARIMA models:
ARIMA stands for Auto-Regressive Integrated Moving Average. ARIMA models are, in theory, the most general class of 
models for forecasting a time series which can be made to be 
“stationary” by differencing (if necessary), perhaps in conjunction with nonlinear transformations such as logging 
or deflating (if necessary). A random variable that is a time series is stationary if its statistical properties 
are all constant over time.  A stationary series has no trend, its variations around its mean have a constant 
amplitude, and it wiggles in a consistent fashion, i.e., its short-term random time patterns always look the same 
in a statistical sense.  The latter condition means that its autocorrelations (correlations with its own prior 
deviations from the mean) remain constant over time, or equivalently, that its power spectrum remains constant 
over time.  A random variable of this form can be viewed (as usual) as a combination of signal and noise, and 
the signal (if one is apparent) could be a pattern of fast or slow mean reversion, or sinusoidal oscillation, 
or rapid alternation in sign, and it could also have a seasonal component.  An ARIMA model can be viewed as a
 “filter” that tries to separate the signal from the noise, and the signal is then extrapolated into the future 
 to obtain forecasts.
 In our case, Non-Stationarity is rejected by the results of Augmented Dickey Fuller (ADF) : p-value of 0.01 -> no clear trend
### ACF: Sample Autocorrelation Function
We first started by generating the ACF graph to help us select the model based on what we could perceive from teh stationarity
as well as the lags that are autocorrelated. As can be seen in the following plots, there is no statistically significant autocrellation between
the Forex lags (without the events data as xreg).
<center><img src="/assets/week_4/ACF.PNG" width="70%"></center>

##Training The Model:
As explained in previous weeks, we trained the model on the log-return data of the USD/GBP 
exchange rates with our best performing heuristic values as an Exogenous variable.
We eliminated linearly dependent columns in the heuristic matrix first and then added
it to the ARIMA model.
<center><img src="/assets/week_4/Residuals.PNG" width="125%"></center>

We sought to find the ARIMA(p,q,d) parameters that minimize our AIC(max log-lik) in order to figure whether our model
had a moving average, multiple lags or non-stationarity (should be able to discover seasonalit as well). It turned out
that no smoothing/differntiating was needed (q=0) and that we could have 2 moving averages.
However, using auto.arima() kept insisting that the best model would be one with only 1 lag and no moving averages 
(which resulted in lower log-lik from our own optimization).
<center><img src="/assets/week_4/2MA.PNG" width="125%"></center>

## Testing: Forecast
ARIMA(2,0,1) model forecast:
<center><img src="/assets/week_4/201 model.PNG" width="125%"></center>
<center><img src="/assets/week_4/30MA.PNG" width="125%"></center>
<center><img src="/assets/week_4/50Day.PNG" width="125%"></center>

## Limitations/Possible Improvements:
The window we use for the training and testing of the autocorrelation of the time series matters a lot since there could besome trends
that only show on an inter-day level whereas others are clear on a multi-year scale. Therefore, we need to define our modeling/forecasting
windows in order to analyze the seasonality/stationarity of the data




