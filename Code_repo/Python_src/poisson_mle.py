import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.misc import factorial


def poisson(k, lamb):
    """poisson pdf, parameter lamb is the fit parameter"""
    return (lamb**k/factorial(k)) * np.exp(-lamb)


def pareto(x, alpha, x_m=1):
    """pareto pdf, parameter alpha is the fit parameter"""
    return 1.0 * alpha * math.pow(x_m, alpha) / np.array(map(lambda y: math.pow(y, alpha + 1), x))


def negLogLikelihoodPoisson(params, data):
    """ the negative log-Likelohood-Function"""
    lnl = - np.sum(np.log(poisson(data, params[0])))
    return lnl


def negLogLikelihoodPareto(params, data):
    """ the negative log-Likelohood-Function"""
    lnl = - np.sum(np.log(pareto(data, params[0])))
    return lnl


# get poisson deviated random numbers
data = np.random.pareto(0.3, 1000)

# minimize the negative log-Likelihood

result = minimize(negLogLikelihoodPareto,  # function to minimize
                  x0=np.ones(1),     # start value
                  args=(data,),      # additional arguments for function
                  bounds=(0, None),
                  method='SLSQP',   # minimization method, see docs
                  )
# result is a scipy optimize result object, the fit parameters 
# are stored in result.x
print(result)

# plot poisson-deviation with fitted parameter
x_plot = np.linspace(1, 20, 1000)

plt.hist(data, bins=np.arange(15) - 0.5, normed=True)
plt.plot(x_plot, pareto(x_plot, result.x), 'r-', lw=2)
plt.show()