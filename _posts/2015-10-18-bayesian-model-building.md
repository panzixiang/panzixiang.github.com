---
layout: post
title: "Bayesian model building"
description: ""
category: 
tags: []
---
{% include JB/setup %}


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
<center><img src="/assets/week_4/Residuals.PNG" width="70%"></center>

We sought to find the ARIMA(p,q,d) parameters that minimize our AIC(max log-lik) in order to figure whether our model
had a moving average, multiple lags or non-stationarity (should be able to discover seasonalit as well). It turned out
that no smoothing/differntiating was needed (q=0) and that we could have 2 moving averages.
However, using auto.arima() kept insisting that the best model would be one with only 1 lag and no moving averages 
(which resulted in lower log-lik from our own optimization).
<center><img src="/assets/week_4/2MA.PNG" width="70%"></center>

## Testing: Forecast
ARIMA(2,0,1) model forecast:
<center><img src="/assets/week_4/201 model.PNG" width="70%"></center>
<center><img src="/assets/week_4/30MA.PNG" width="70%"></center>
<center><img src="/assets/week_4/50DayMA.PNG" width="70%"></center>

## Limitations/Possible Improvements:
The window we use for the training and testing of the autocorrelation of the time series matters a lot since there could besome trends
that only show on an inter-day level whereas others are clear on a multi-year scale. Therefore, we need to define our modeling/forecasting
windows in order to analyze the seasonality/stationarity of the data

