import csv
import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import time

from sklearn import gaussian_process


_DATE_COLUMN = 0
_RATE_COLUMN = 1


def f(x):
    return x * np.sin(x)


def load_csv(filename):
    """
    :return: list of rows (where each row is a list)
    """
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        return list(reader)


def get_rate_column(raw_data):
    """
    Get the rates column from raw_data
    :param raw_data: list of rows
    :return: list
    """
    return np.array(map(lambda row: float(row[_RATE_COLUMN]), raw_data))


def get_date_column(raw_data):
    """
    Get the dates column from raw_data
    :param raw_data: list of rows
    :return: list
    """
    return np.array(map(lambda row: float(row[_DATE_COLUMN]), raw_data))


if __name__ == "__main__":
    # Read raw input
    input_file = 'gbpusd_2005.csv'
    raw_data = load_csv(input_file)[1:]
    rates = get_rate_column(raw_data)
    mean_rate = sum(rates) * 1.0 / len(rates)
    rates = rates - mean_rate # Make the mean zero
    days = np.array(xrange(len(rates)))

    # Use the past 30 days' rates to predict the next day
    num_days_to_predict = 360
    correct_prediction = [0.0] * num_days_to_predict

    for base_day in xrange(0, num_days_to_predict):
        # Prepare the training and the test data
        num_train = 3
        num_test = 1
        days_train = np.atleast_2d(days[base_day : base_day + num_train]).T
        rates_train = rates[base_day : base_day + num_train].ravel()
        days_test = np.atleast_2d(days[base_day + num_train: base_day + num_train + num_test]).T
        rates_test = rates[base_day + num_train: base_day + num_train + num_test].ravel()

        # Train and fit the regression
        gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1, corr="linear")
        gp.fit(days_train, rates_train)

        rates_pred, sigma2_pred = gp.predict(days_test, eval_MSE=True)
        sigma_pred = map(lambda s2: math.sqrt(s2), sigma2_pred)
        upper_pred = map(lambda a, b: a + 1.96 * b, rates_pred, sigma_pred)
        lower_pred = map(lambda a, b: a - 1.96 * b, rates_pred, sigma_pred)

        if (rates_pred[0] > rates_train[-1] and rates_test[0] > rates_train[-1]) or \
            (rates_pred[0] < rates_train[-1] and rates_test[0] < rates_train[-1]):
            correct_prediction[base_day] = 1.0

    print sum(correct_prediction)

    if False:
        # Show results
        line1, = plt.plot(days_train, map(lambda x: x + mean_rate, rates_train), 'b-', lw=2, label="training")
        line2, = plt.plot(days_test, map(lambda x: x + mean_rate, rates_test), 'r-', lw=2, label="test label")
        line3, = plt.plot(days_test, map(lambda x: x + mean_rate, rates_pred), 'g-', lw=2, label="test prediction")
        line4, = plt.plot(days_test, map(lambda x: x + mean_rate, upper_pred), 'g--', lw=2)
        plt.plot(days_test, map(lambda x: x + mean_rate, lower_pred), 'g--', lw=2)
        plt.ylabel("GBP-USD exchange rate")
        plt.xlabel("Day of the year")
        plt.title("GBP-USD Exchange Rate from 2005")
        plt.legend([line1, line2, line3, line4], ["Training", "Test label", "Test prediction", "95% prediction interval"], loc=3)
        plt.show()


# Binary classification accuracy using 30 days of training
# Squared exponential correlation function: 146 / 330
# absolute_exponential: 160 / 330
# linear: 149 / 330

# 60 days of training
# Squared exponential correlation function: 140 / 330
# absolute_exponential: 139 / 330
# linear: 141 / 330

# 10 days of training
# Squared exponential correlation function: 153 / 330
# absolute_exponential: 169 / 330
# linear: 156 / 330

# 3 days of training
# Squared exponential correlation function: 164 / 330
# absolute_exponential: 159 / 330
# linear: 135 / 330



