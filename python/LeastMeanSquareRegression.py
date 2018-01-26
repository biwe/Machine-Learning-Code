import matplotlib.pyplot as plt
import numpy as np


def least_mean_square_batch(x, y):
    """This is a practical least mean square regression
	in batch mode"""
    theta0_pre, theta1_pre = 0, 0
    theta0, theta1 = 1, 1
    alpha = 0.01
    con = 0.1**5
    while np.fabs(theta1 - theta1_pre) > con or np.fabs(theta0 - theta0_pre) > con:
        error_0 = np.sum((y - theta0 - theta1 * x) * 1)/len(y)
        error_1 = np.sum((y - theta0 - theta1 * x) * x)/len(y)
        theta0_pre = theta0
        theta1_pre = theta1
        theta0 = theta0_pre + alpha * error_0
        theta1 = theta1_pre + alpha * error_1

    return theta0, theta1


def least_mean_square_normal(x, y):
    """This is a practical least mean square regression for normal"""
    X = np.c_[np.ones(len(x)), x]
    theta = np.linalg.inv(X.transpose() @ X) @ X.transpose() @ y

    return theta[0], theta[1]


x1 = np.arange(1000)*0.01
y1 = x1 + np.random.rand(1000) - 1

theta_0, theta_1 = least_mean_square_batch(x1, y1)

print(theta_1)

plt.plot(x1, y1, 'g-')
plt.plot(x1, theta_0+theta_1*x1, 'r-')

plt.show()
