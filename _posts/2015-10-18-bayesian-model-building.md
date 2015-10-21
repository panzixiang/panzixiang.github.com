---
layout: post
title: "Bayesian model building"
description: ""
category: 
tags: []
---
{% include JB/setup %}



## Improving Heuristics for GDELT Feature Engineering

Section on Pareto/Poisson distribution fitting


## Gaussian Process Regression

Section on Gaussian Process



We performed Gaussian Process Regression on the historical exchange rates between GBP and USD in 2005. In the first image below, we used the first 40 days of 2005 as the training data to predict the next 10 days' exchange rates.


<center><img src="/assets/week_4/gaussian_process_40_days.png" width="100%"></center>

In the second image, we used the first 50 days of 2005 as the training data to predict the next 10 days' exchange rates.

<center><img src="/assets/week_4/gaussian_process_50_days.png" width="100%"></center>

In the second image, we used the first 70 days of 2005 as the training data to predict the next 10 days' exchange rates.

<center><img src="/assets/week_4/gaussian_process_70_days.png" width="100%"></center>

We use Gaussian Process Regression to frame this as a binary classification problem: use the historical prices to predict the mean of the next day's price:

$$ prediction(T+1) = E[F_{T+1}| F_1, ... , F_T] $$

If the prediction is more than the current price, predict an upward trend; otherwise, predict a downward trend.

The table below shows the classification accuracy of running kernel SVM with the different heuristics on the test set:

|Number of training days         | Best correlation function  | Test accuracy |
|:------------------------:|:-----------------|:----------------|
| 60 | Linear   | 47.0%   |    
| 30 | Absolute exponential   | 48.0%   |
| 10 | Absolute exponential  | 48.3%   |
| 3 | Squared exponential  | 49.7%   |



## Vector Autoregressive Model

Ghassen's section




