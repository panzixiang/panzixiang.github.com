---
layout: post
title: "Week 2 Literature review and methods analysis"
description: "Some initial explorations"
category: class-posts
tags: [literature review, data]
---
{% include JB/setup %}


## Goal 

- Condense the GDELT data columns by removing redundancy and aggregating similar 	features
- Impose a function to change domain from discrete/categorical variables into the real numbers.
- Devise a testable method to explore various signals in the financial markets (price, volume and volatility of various stocks, indices and foreign exchange rates)


## Literature Review

We have reviewed a list of papers that explored methods in dealing with the GDELT dataset as well as some dimensionality reduction models.

- [Event-based stock market prediction](http://cs229.stanford.edu/proj2014/Hadi%20Pouransari,%20Hamid%20Chalabi,%20Event-based%20stock%20market%20prediction.pdf)
	- Key concept: The authors tried to predict whether a stock’s price will rise in the day after its earnings announcement has been published. The two sources of data that were used are: historical prices of the stock and its earnings statements. 54 numerical features were extracted from the two sources of data for binary classification.
	- Successes/Improvements: The approach using logistic regression with regularization achieved a test error of 36.1%. A SVM with 3rd degree polynomial kernel achieved a test error of 36.0%. An extension to their work would be to predict the change in price as a continuous variable instead of a binary variable. Training a separate model for each sector might also improve the performance.

<figure>
	<figcaption>Training and Test Errors for Different SVM Kernels</figcaption>
	<img src="/assets/week_2/SVM_Kernel.png" width="95%">
</figure>


- [Using Structured Events to Predict Stock Price Movement: An Empirical Investigation]( http://emnlp2014.org/papers/pdf/EMNLP2014148.pdf)
- [Deep Learning for Event-Driven Stock Prediction](http://ijcai.org/papers15/Papers/IJCAI15-329.pdf)
	- Key concepts: The Ding et al. deep learning papers use natural language processing along with deep learning to perform binary classification on stocks’ movements. First, a structured representation including the subject, action, object, etc is extracted from news articles from Reuters and Bloomberg into a feature vector. Then, Ding et al. experimented with various neural network architectures in the different papers.

	- Successes/Improvements: Their latest architecture achieved 64% accuracy for the S&P 500 index and 65% accuracy for individual stocks. The strength of this architecture is that its model combines the short-term effects and the long-term effects of news on the market. This architecture is as follows: the aggregated event embeddings for each day serve as input to the NN. Events from the current day are treated as short term events; events from the past week are mid-term events; events from the past month are long-term events. There are separate convolution layers for long-term events and mid-term events. The output layer only has two neurons which predicts whether the stock price will go up or go down.

	<img src="/assets/week_2/Deep_Learning_for_Stocks.png" width="100%">


- [Stock Prediction Using Event-based Sentiment Analysis](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6690034)

- [Predicting the Present With Google Trends](http://people.ischool.berkeley.edu/~hal/Papers/2011/ptp.pdf)
	- Key concepts: Google Trends is an index of the volume of queries that users enter into Google. It measures the total query volume for a particular search term within a particular geographic region relative to the total query volume in that region. Even though it is only available to the public with a weekly granularity, Google Trends data can still provide insights of the current state of the economy. In Predicting the Present with Google Trends, Hal Varian constructed autoregressive models based on Google Trends data in order to infer economic indicators such as automobile sales and number of tourists in the US.
	- Successes/Improvements: Simple AR models that include relevant Google Trends data tend to outperform models that exclude these data by 5% to 20%. For example, Varian used Google Trends data to augment the simple AR model which could model motor vehicle sales in the US 10.5% better than the original model.

	<img src="/assets/week_2/Motor_Vehicle_Google_Trends.png" width="100%">

- [Stock Prices, News, and Business Conditions](http://www.nber.org/papers/w3520.pdf)
- [The Three-Pass Regression Filter](http://faculty.chicagobooth.edu/bryan.kelly/research/pdf/Forecasting_theory.pdf)
- [Isomap](http://wearables.cc.gatech.edu/paper_of_week/isomap.pdf)
	- Key concepts: 
		- Isomap inherits from PCA and MDS: a noniterative, polynomial time procedure with a guarantee of global optimality; for intrinsically Euclidean manifolds, a guarantee of asymptotic conver-gence to the true structure.
		- Isomap is applicable in mapping highly categorical data as long as we have a reasonable metric and paths on a graph/network of events/datapoints.
- [Visual and Predictive Analyticson Singapore News: Experiments on GDELT, Wikipedia, and ˆSTI](http://arxiv.org/pdf/1404.1996v1.pdf)
    - See the feature engineering section for details


## Feature Engineering on GDELT data


### Feature representation

The paper on using GDELT to analyze Singapore's stock proposed a way to convert the features of the GDELT data from categorical data to numerical data. Their approach is as follows: If a column V has categorical data, first find the set of unique categories d = {d1, d2, …, dk}. Then convert this column into k different columns V_1, V_2, …, V_k with one-hot encoding. The data points from each day is aggregated by summing the one-hot encoded vectors. Note that for continuous variable, the one hot encoding represents different bins that the data falls into. Below is a dummy example of the transformations:

<img src="/assets/week_2/GDELT_Original_Format.png" width="60%" align="middle">
<img src="/assets/week_2/GDELT_One_Hot_Encoding.png" width="90%" align="middle">
<img src="/assets/week_2/GDELT_Aggregated_by_Date.png" width="90%" align="middle">

We will use the above feature engineering method to aggregate our data by the day: first converting each line into a one-hot encoded vector, then summing all vectors within the same day to obtain daily statistics. Because this approach will introduce a significant number of dimensions, it is impractical to run this algorithm on all the columns. We will instead focus on a small subset of the columns which are relevant to our purposes:


| Column                           | Description  |
| :--------------------------------|:-------------|
| Actor1Code, Actor2Code | Identification of actors |
| EventCode              | Hierarchical CAMEO code for event classification |
| QuadClass              | Material/Verbal Conflict/Cooperation classification |
| GoldsteinScale         | Captures the theoretical impact on the stability of a country |
| NumMentions            | Proxy for the impact of the event |
| AvgTone                | How positive/negative the news article's tone is |


This has a few advantages:

- It reduces the number of dimensions we have to consider my aggregating them onto one scale while maintaining information about the event impact, which is what we are ultimately concerned about.
- We can fine-tune the degree at which each column above "proxy" the output value, such we can make rapid changes at low cost.


### Isomap
We are making an initial assumption that the actors and event impact are independent, i.e. that the magnitude of the impact of events will be likely the same regardless of who perpetrated that. Thus we can map the two actors of an events onto an isomap with the distance metric being the number of times they were mentioned together: 

> A metric on a set $$X$$ is a function $$d: X \times X \rightarrow \mathbb{R}$$ with the following conditions:
> 
> Non-negativity + coincidence: $$d(x,y)\geq 0$$ with equality only at $$d(x,x)$$
>
> Symmetry: $$d(x,y)=d(y,x)$$
>
> Triangle inequality: $$d(x,z)\leq d(x,y)+d(y,z)$$ 

We devise a function that takes the number of mentions of both actors in a row of GDELT data table and apply a function (such as the logarithm) to map it to the positive reals. Thus we can construct a graph with vertices being actors and edge distances proportional the "connectedness" (and inversely proportional) to the number of joint mentions of them.This will give us a structure to apply the isomap on and seek a low-dimensional manifold that encompass relationships between actors.

<img src="/assets/isomap.png" width="100%">

## Potential Probabilistic Models

### Vector Autoregressive Models (VAR) on Financial Time Series
VAR models are some of the most flexible and easy to use models for multivariate time series. It has proven to be especially useful for describing the dynamic behavior of economic and financial time series and
for forecasting.
Let $$Y_t = (y_1, y_2,...,y_nt)'$$ denote an $$(n \times 1)$$ vector of time series variables. The basic p-lag vector autoregressive model (VAR(p)) has the form:

$$\bf{Y}_t=\bf{c}+\bf{\Pi}_1\bf{Y}_{t-1}+\bf{\Pi}_2\bf{Y}_{t-2}+\cdots+\bf{\Pi}_p\bf{Y}_{t-p}+\epsilon_t, \ \ t=1\ldots T$$

where $$\Pi_i$$ are $$(n \times n)$$ matrix and $$\epsilon_t$$ is an $$(n \times 1)$$ unobservable zero mean noise vector process.

#### Exogenous Factors in VAR
To include the exogenous factors into a VAR model (in this case: a vector constructed from our events database), the VAR form can be extended in the following manner:

$$\bf{Y}_t=\bf{c}+\bf{\Pi}_1\bf{Y}_{t-1}+\bf{\Pi}_2\bf{Y}_{t-2}+\cdots+\bf{\Pi}_p\bf{Y}_{t-p}+\bf{\Phi D}_t+\bf{GX}_t+\epsilon_t$$

where $$D_t$$ represents an $$(l \times l)$$ matrix of deterministic components, $$X_t$$ represents an $$(m \times 1)$$ matrix of exogenous variable and $$\Phi$$ and $$G$$ are parameter matrices.

$$\bf{y}_t=A^{-1}A^*_1\bf{y}_{t-1}+\ldots+A^{-1}A^*_p\bf{y}_{t-p}+A^{-1}A^*_1B\epsilon_t$$
$$\bf{y}_t=A^{-1}A^*_1\bf{y}_{t-1}+\ldots+A^{-1}A^*_p\bf{y}_{t-p}+\bf{u}_t$$

#### Identifying Exogenous Events as an Impluse Response in VAR
Since the contribution of Sims (1980) the interaction between variables and disturbances in VAR models has been best described and interpreted by impluse response functions.
<img src="/assets/week_2/Impact.PNG" width="100%" align="middle">
An impulse resonse is when a shock is assigned to one variable of the system and where the propagation of this shock on all the variables of the system s studied over time.
Impulse response functions are used to describe how the economy reacts over time to exogenous impulses, which economists usually call shocks. Impulse response functions describe the reaction of endogenous macroeconomic variables at the time of the shock and over subsequent points in time.
The standard method to identify such shocks is through recursive identification where we impose a certain ordering to the variables, hence assuming that all contemporaneous interactions among variables are recursive. This corresponds to a B model allowing for instantaneous effects of the shocks on the variables which can be written as follows:

$$y_t=A^{(0)}_{s_t}+\sum_{i=1}^p A^{(i)}_{s_t}y_{t-i}+B_{s_t}\epsilon_t$$

$$\epsilon_t \sim i.i.\mathcal{N}(0,I_N)$$

#### Markov Switching VAR: Non-Linear Model for Different Regimes of the economy:
The Markov-switching vector autoregression (MSVAR) was introduced by Hamilton (1989) and has proved useful in capturing important non-linearities in economic processes.
A regime $$s_t$$ is assumed to follow a hidden m-state Markov-chain. The probability of being in regime j next period conditional on the current regime $$i$$ is assumed exogenous and constant.

Many financial time series processes appear subject to periodic structural changes in their dynamics. Regression relationships are often not robust to outliers nor stable over time, whilst the existence of changes in variance over time is well documented


There has been recent work on deriving impulse responses for Markov-Switching VAR models by taking into account the history of the system (times eries) as well as the type of the shocks impacting it.
Once those structural shocks are identified. We could seek to find a correlation between them and the vectors of events from GDELT.

#### Estimating a Markov-Swtiching VAR:
A Bayesian framework is developed to estimate the relations between the regimes, the time series and the impulse responses.
A Bayesian Markov Chain Monte Carlo estimation procedure is developed which generates the joint posterior density of the parameters and the regimes, rather than the more common point estimates. The complete likelihood surface is generated at the same time. This can be done with a Gibbs sampler that yield the sampled posterior distributions.
Having defined Bayesian impluse responses for MSVAR, Model Selection can be done using Kullback-Leiber divergene that uses the marginal density of a given model. This helps with picking Linear vs. Non-Linear models as well as selecting the number of lags.
Finally, Bayesian estimation, with the possibility of imcorporating the prior, can help cirvumvent the curse of dimenstionality using Bayesian Shrinkage: assignging a low a priori to parameters thought to play a less important role.

Accordingly, Bayesian inference yielding posterior densities instead of point parameters, inference on the likeliness of the responses can immediately be performed from their posteriors. THis is important because nonlinear moels with switching regimes have much richer dynamics than linear ones and the range of possible impulse responses can become large. 

#### Lag Length Selection: How many days to go back in history
The lag length for the VAR(p) model may be determined using model selection criteria. The general approach is to fit VAR(p) models with possible values of p which minimizes some model selection critiera such as Akaike (AIC), Schwarz-Bayesian(BIC) and Hannan-Quinn(HQ) based on the size of the features n and the residual  covariance matrix:

$$\Sigma^~(p)=T^{-1}\sum^T_{t=1}\hat{\epsilon}_t\hat{\epsilon}'_t$$

$$AIC(p)=\ln{|\Sigma^~(p)|}+\frac{2}{T}pn^2$$

$$BIC(p)=\ln{|\Sigma^~(p)|}+\frac{\ln{T}}{T}pn^2$$

$$HQ(p)=\ln{|\Sigma^~(p)|}+\frac{2\ln{\ln{T}}}{T}pn^2$$

### Hidden Markov Models:
Time series can be modeled as a geometric Brownian motion with drift. Especially, in financial engineering field, the stock model, which is also modeled as geometric
Brownian motion, is widely used for modeling derivatives:
<img src="/assets/week_2/eq9.PNG" width="50%" align="middle">
Here, the coefficients of the drift, $$\mu$$ and volatility $$\sigma$$ are constant. However, in a Bear marke (Internet Bubble) or Bull market (obvious economic growth), it is difficult to discern which situation we are in and the mean/variance of the stock will be totally different.
Therefore, we need to define regions of time as a regime whose mean and variacne are expliclity different from other regions of time. A reg

Sotck returns series would then be modelled as a mixture of Gaussian distribution and discrete Markov chain. In a certain regime, stock series follow geometric Brownian motion with drift, which means stock returns have Gaussian distribution and the regimes are changing by the discrete Markov Process.
For example, while, in good economic situation, stock return has higher mean and smaller variance, it has lower mean and larger variance in bad economic situation. In this case, we have two economic states, i.e. good and bad, and each state has different mean and variance.
The regime changing occurs randomly in this model. However, the changing probability is fixed and consistent thought the stock return series.

* HMMs can be used to divide the entire time series into different regimes(each characterized by a specific volatility level, for example).
* The return of the time series is assumed to be modelled by a mixture of probability densiteis and each density corresponds to a hidden state with its mean and variancce.
* Viterbi algorithm in HMMs can be used to find the state sequence in the time series. We then get the subsets of original time series corresponding to different states.
* Another option would be to use Expectation Maximization (EM) to maximize the likelihood of historical returns based on the mixture of Gaussian and discrete Markoc Chain:
 * $$X_t$$ is the state of Mean and variance model at time t.
 * $$R_t$$ is the stock return at time t.
 <img src="/assets/week_2/eq10.PNG" width="50%" align="middle">
 * EM seeks to maximise the posterior Likelihood as below:
 <img src="/assets/week_2/eq11.PNG" width="50%" align="middle">
* We could also use Double HMM which runs the Markov Chain of the economic states separately which gives the model more degrees of freedom.
* We would then seek the correlation between the change of regimes and our shocks/events.

### Kalman Filters:
Similarily to HMM models, the theory of finance is mainly treated in term of stochastic differential equations such as, the value of a stock price S is supposed to follow a diffusion geometrical Wiener process which incorporates a drift function as well as the volatilty of the stock price.
The Kalman filter (KF) can not be used for this analysis since the functions are nonlinear and the transition density of the state space is non-Gaussian. But with the advent of new estimation methods such as Markov Chain Monte Carlo (MCMC) and Particle filters (PF), exact estimation tools for nonlinear state-space and non-Gaussian random variables became available.
Unlike a simple moving average hat has a fixed set of windowing parameters, the kalman filter constantly updates the information to produce adaptive filtering on the fly

 <img src="/assets/week_2/kalma.PNG" width="150%" align="middle">

## Performance Validation

### Binary classification
The simplest way to validate the performance is to frame the problem as a binary classification problem: whether the price of stock or the rate of foreign exchange will rise or fall on a particular day. The classification error serves as a metric of the accuracy of prediction.

- The current benchmark for the accuracy for stock/index binary classification is around 65%

### Mock trading
Another way to test models is to devise a simple trading strategy and test it against market performance. For example a strategy might be buying stocks if the model probability of the price going up was higher than a certain threshold and shorting them if the model probability of the price going down was higher than a certain threshold.

- The cumulative return of S&P 500 can be used as a benchmark when evaluating our mock trading strategy