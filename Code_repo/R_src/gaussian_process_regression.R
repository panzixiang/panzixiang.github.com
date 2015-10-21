library(ggplot2)
library(dplyr)
library(tidyr)
library(lattice)
library(quantmod)
# Library for SVM
library(kernlab)
# Library for Gaussian Process Regression 
library(GPFDA)
library(MASS)

setwd('/Users/tomwu/Google Drive/COS513 Project Folder')

# Reading the forex data
forex_data <- read.csv("gbpusd_2005.csv", header=TRUE, sep=',') # read data
forex_data <- as.data.frame(forex_data)
forex_data$Date <- as.Date(as.character(forex_data$Date), "%Y%m%d")
forex_data_ts <- xts(forex_data[,-1], order.by=as.Date(forex_data$Date))
names(forex_data_ts) <- c('Rate')
chartSeries(forex_data_ts, name="GBP USD Forex")

# Extract the date column and the rate columns for processing
rates <- forex_data_ts$Rate
rates <- data.matrix(as.data.frame.ts(rates))
dates <- forex_data$Date
dates <- data.matrix(as.data.frame.ts(dates))

# GPR
hyperparams <- list("pow.ex.w"=log(1000),"linear.a"=log(1000),"pow.ex.v"=log(500),"vv"=log(100))
num_train <- round(nrow(rates) * 0.1)
rates_train <- rates[1:num_train]
dates_train <- dates[1:num_train]
rates_test <- rates[(num_train + 1) : (num_train + 10)]
dates_test <- dates[(num_train + 1) : (num_train + 10)]

regression <- gpr(dates_train, rates_train, c("linear","pow.ex"), hyperparams, trace=2)

rates_predict <- gppredict(regression, dates_test)

plot(dates_train, rates_train, type="l", col="red", 
     xlim=c(min(dates_train), max(dates_test)),
     ylim=c(min(rates_train, rates_test), max(rates_train, rates_test)))
lines(dates_test, rates_test, type="l", col="blue")
lines(dates_test, rates_predict$pred.mean, type="l", col="green")


