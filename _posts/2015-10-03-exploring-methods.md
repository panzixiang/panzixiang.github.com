---
layout: post
category : class-posts
tagline:
tags : [exploratory-analysis,data]
---
{% include JB/setup %}

# Week 2: Exploratory analysis

## Our Goal (to reiterate)

We have the raw news data time series:
$$
\{r_n\}_{n=0}^{M}, r_t\in R
$$, where $$R$$ is a 54-tuple over mostly categorical variables and we have $$M$$ news history.

We also have $$\{x_t^{(i)}\}_{t=t_0-N}^{t_0}$$, the time series data for EOD commodity price of the $$(i)$$-th commodity on the $$t$$-th day, in $$\mathbb{R}_+$$.

Initially, we'll be trying just to find, for a given $$i$$, $$f(\{r_n\}_{n=0}^{M},\{\textbf{x}_{t=t_0-N}^{t_0}\})\approx \mathbb{P}\left(x_{t+1}^{(i)}>x_{t}^{(i)}\right)$$.

## Previous work on GDELT

http://people.cs.vt.edu/naren/papers/websci-gdelt-2014.pdf
??

## Approach

1. Previous exploration has indicated there is some aspect of lower dimenality in $$R$$ - agents would only interact with a small subset of other agents, each agent would only be affected by small subsets of types of events. Thus we'd like to reduce from $$R$$ to $$\mathbb{R}^d$$ for $$d < 54$$.
2. Evaluate what features we can extract from $$\{\textbf{x}_{t=t_0-N}^{t_0}\}$$ alone - according to previous readings, summary statistics, such as moving averages, have some predictive power.
3. Develop our forecasting model - consider the various regression methods for finding $$f$$.

## Part 1: Dimensionality Reduction

{% include 2015-10-03-includes/dimensionality_reduction.md %}

## Parts 2 and 3: Forecasting

{% include 2015-10-03-includes/forecasting.md %}
