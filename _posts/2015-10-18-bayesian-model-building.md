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

$$ H_1 = NumArticles(non-normalized) \times GoldSteinScale $$

Preliminaries show no linear relationship between regressors and predictors.

#### Second heuristic

$$ H_2 = NumArticles(non-normalized) \times AvgTone $$

#### Third heuristic

$$ H_3 = NumArticles(non-normalized) \times QuadClass $$

One possible approach is to normalize the number of articles by the total number of articles in the same day. We tried fitting the number of articles within each day



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


## Vector Autoregressive Model

Ghassen's section




