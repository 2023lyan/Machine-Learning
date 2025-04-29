import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    x_eval, _ = util.load_dataset(eval_path, add_intercept=False)
    clf = GDA()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_eval)
    np.savetxt(pred_path, y_pred.T > 0.5, fmt="%d")
    util.plot(x_train, y_train, clf.theta, f"output/p01e_gda_{pred_path[-5]}")
    # *** END CODE HERE ***


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
        self.theta = np.zeros(n + 1)
        self.theta[0] = theta_0
        self.theta[1:] = theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        theta_0 = self.theta[0]
        theta = self.theta[1:].reshape(-1, 1)
        return 1 / (1 + np.exp(- theta.T.dot(x.T) - theta_0))
        # *** END CODE HERE
