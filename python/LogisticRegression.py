import matplotlib.pyplot as plt
import numpy as np


def logistic_regression_batch(x, y):
    """This is a practical least mean square regression
    x, y are 1*n array, respectively"""
    theta0_pre, theta1_pre = 0, 0
    theta0, theta1 = 1, 1
    alpha = 0.01
    con = 0.1**4
    while np.fabs(theta1 - theta1_pre) > con or np.fabs(theta0 - theta0_pre) > con:
        hx = 1/(1 + np.exp(-(theta0 + theta1 * x)))
        error_0 = np.sum((y - hx) * 1)/len(y)
        error_1 = np.sum((y - hx) * x)/len(y)
        theta0_pre = theta0
        theta1_pre = theta1
        theta0 = theta0_pre + alpha * error_0
        theta1 = theta1_pre + alpha * error_1

    return theta0, theta1


x1 = np.arange(100)
x1 = x1[:, np.newaxis]
y1 = np.r_[np.zeros(20), np.ones(80)]
y1 = y1[:, np.newaxis]

theta_0, theta_1 = logistic_regression_batch(x1, y1)

plt.scatter(x1, y1)
plt.plot(x1, 1/(1 + np.exp(-(theta_0 + theta_1 * x1))), 'r-')

plt.show()
