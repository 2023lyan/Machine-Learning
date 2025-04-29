import numpy as np
import util

from linear_model import LinearModel
from scipy.stats import boxcox


def transformation(x, lam):
    m, n = x.shape
    x_new = np.zeros((m, n))
    if set(lam).issubset({0}):
        for i in range(n):
            u = x[:, i]
            u_trans, lam[i] = boxcox(u + 1)
            x_new[:, i] = u_trans
    else:
        for i in range(n):
            u = x[:, i]
            u_trans = boxcox(u + 1, lmbda = lam[i])
            x_new[:, i] = u_trans
    return x_new, lam

class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        x, lam = transformation(x, lam = np.zeros(n)) # do the box-cox transformation
        y1_num = sum(y == 1)
        y0_num = m - y1_num
        phi = y1_num / m
        mu_0, mu_1 = 0, 0
        for i in range(m):
            mu_0 = mu_0 + (1 - y[i]) * x[i, :]
            mu_1 = mu_1 + y[i] * x[i, :]
        mu_0 /= y0_num
        mu_1 /= y1_num
        Sigma = np.zeros((n, n))
        for i in range(m):
            mu = mu_1 * y[i] + mu_0 * (1 - y[i])
            u = x[i, :] - mu
            u = u.reshape(-1, 1)
            Sigma = Sigma + 1 / m * np.matmul(u, u.T)
        mu_0 = mu_0.reshape(-1, 1)
        mu_1 = mu_1.reshape(-1, 1)
        Sigma_inv = np.linalg.inv(Sigma)
        theta = Sigma_inv.dot(mu_1 - mu_0)
        theta_0 = 1 / 2 * (np.matmul(mu_0.T, np.matmul(Sigma_inv, mu_0)) - np.matmul(mu_1.T, np.matmul(Sigma_inv, mu_1))) + np.log(phi / (1 - phi))
        theta_0 = theta_0.reshape(-1)
        theta = theta.reshape(-1)
        print(theta)
        self.theta = np.zeros(2 * n + 1)
        self.theta[0: n] = lam
        self.theta[n] = theta_0
        self.theta[n + 1:] = theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        _, n = x.shape
        x, _ = transformation(x, lam = self.theta[0: n]) # do the box-cox transformation
        theta_0 = self.theta[n]
        theta = self.theta[n + 1:].reshape(-1, 1)
        return 1 / (1 + np.exp(- theta.T.dot(x.T) - theta_0))
        # *** END CODE HERE

x_train, y_train = util.load_dataset('../data/ds1_train.csv', add_intercept=False)
x_eval, _ = util.load_dataset('../data/ds1_valid.csv', add_intercept=False)
clf = GDA()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_eval)
np.savetxt('output/p01e_box_cox_1.txt', y_pred.T > 0.5, fmt="%d")
_, num = x_train.shape
x_train, _ = transformation(x_train, lam = np.zeros(num))
util.plot(x_train, y_train, clf.theta[num:], f"output/p01e_gda_box_cox_1")