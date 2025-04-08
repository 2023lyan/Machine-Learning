import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_eval, _ = util.load_dataset(eval_path, add_intercept=True)
    clf = LogisticRegression(eps=1e-5)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_eval)
    np.savetxt(pred_path, y_pred, fmt="%d")
    util.plot(x_train, y_train, clf.theta, f"output/p01b_pred_{pred_path[-5]}")
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def g(self, z):
        return 1 / (1 + np.exp(-z))

    def h(self, theta, x):
        return self.g(np.matmul(theta, x))

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m = x.shape[0]
        n = x.shape[1]
        theta = np.zeros(n)
        def d_l(theta):
            sum_d_l = np.zeros(n)
            for i in range(m):
                sum_d_l = sum_d_l - 1 / m * (y[i] - self.h(theta, x[i,:])) * x[i,:]
            return sum_d_l
        def d_d_l(theta):
            sum_d_d_l = np.zeros((n, n))
            for i in range(m):
                x_i = x[i,:].reshape(-1, 1)
                sum_d_d_l = sum_d_d_l + 1 / m * self.h(theta, x[i,:]) * (1 - self.h(theta, x[i,:])) * np.matmul(x_i, x_i.T)
            return sum_d_d_l
        while True:
            theta_decrease = np.matmul(np.linalg.inv(d_d_l(theta)), d_l(theta))
            epsilon = self.eps
            theta = theta - theta_decrease
            if np.linalg.norm(theta_decrease) < epsilon:
                break
        self.theta = theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return self.g(np.matmul(self.theta, x.T)) > 0.5
        # *** END CODE HERE ***
