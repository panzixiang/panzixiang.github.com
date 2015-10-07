---
layout: post
title: "Week 2 Literature review and methods analysis"
description: ""
category: class-posts
tags: [literature review, data]
---
{% include JB/setup %}


We have done some initial exploration about the GDELT dataset, (insert general comment here)


## Literature review

We have reviewed a list of papers that explored methods in dealing with the GDELT dataset as well as some dimensionality reduction models.

- [Event-based stock market prediction](http://cs229.stanford.edu/proj2014/Hadi%20Pouransari,%20Hamid%20Chalabi,%20Event-based%20stock%20market%20prediction.pdf)
	- Key concept: The authors tried to predict whether a stock’s price will rise in the day after its earnings announcement has been published. The two sources of data that were used are: historical prices of the stock and its earnings statements. 54 numerical features were extracted from the two sources of data for binary classification.
	- Successes/Improvements: The approach using logistic regression with regularization achieved a test error of 36.1%. A SVM with 3rd degree polynomial kernel achieved a test error of 36.0%. An extension to their work would be to predict the change in price as a continuous variable instead of a binary variable. Training a separate model for each sector might also improve the performance.

- [Using Structured Events to Predict Stock Price Movement: An Empirical Investigation]( http://emnlp2014.org/papers/pdf/EMNLP2014148.pdf)
- [Stock Prediction Using Event-based Sentiment Analysis](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6690034)
- [Deep Learning for Event-Driven Stock Prediction](http://ijcai.org/papers15/Papers/IJCAI15-329.pdf)
	- Key concepts: The Ding et al. deep learning papers use NLP to determine actor pairings along with the connecting action (creating event tuples and later event embeddings) and use various neural network architectures for binary classification.  Generally achieved around 60% classification accuracy, both for individual stocks and indices.
	
- [Predicting the Present With Google Trends](http://people.ischool.berkeley.edu/~hal/Papers/2011/ptp.pdf) 
- [Stock Prices, News, and Business Conditions](http://www.nber.org/papers/w3520.pdf)
- [The Three-Pass Regression Filter](http://faculty.chicagobooth.edu/bryan.kelly/research/pdf/Forecasting_theory.pdf)
- [Visual and Predictive Analyticson Singapore News:Experiments on GDELT, Wikipedia, and ˆSTI](http://arxiv.org/pdf/1404.1996v1.pdf)
- [Isomap](http://wearables.cc.gatech.edu/paper_of_week/isomap.pdf)
	- Key concepts: 
		- Isomap inherits from PCA and MDS: a noniterative, polynomial time procedure with a guarantee of global optimality; for intrinsically Euclidean manifolds, a guarantee of asymptotic conver-gence to the true structure.
		- Isomap is applicable in mapping highly categorical data as long as we have a reasonable metric and paths on a graph/network of events/datapoints.

## Performance Validation: some comments

### Binary classification
The simplest way to validate the performance is to frame the problem as a binary classification problem: whether the price of stock or the rate of foreign exchange will rise or fall on a particular day. The classification error serves as a metric of the accuracy of prediction.

A common way to test models is to devise a simple trading strategy and test it against market performance. For example a strategy might be buying stocks if the model probability of the price going up was higher than a certain threshold and shorting them if the model probability of the price going down was higher than a certain threshold.


## Feature representation in GDELT columns - some initial exploration
