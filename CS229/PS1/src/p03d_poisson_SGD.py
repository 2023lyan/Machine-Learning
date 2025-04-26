import numpy as np
import util
import matplotlib.pyplot as plt

from linear_model import LinearModel



class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.theta = np.zeros(n).reshape(-1, 1)
        while True:
            theta = self.theta
            for i in range(m):
                theta = theta + self.step_size * (y[i] - np.exp(x[i, :].reshape(1, -1).dot(theta))) * x[i, :].reshape(-1, 1) / m
            if np.linalg.norm(theta - self.theta, ord = 1) < self.eps:
                self.theta = theta
                break
            else:
                self.theta = theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.exp(x @ self.theta)
        # *** END CODE HERE ***


# Load training set
lr=1e-7,
train_path='../data/ds4_train.csv'
eval_path='../data/ds4_valid.csv'
x_train, y_train = util.load_dataset(train_path, add_intercept=True)
clf = PoissonRegression(step_size=lr, eps=1e-5)
clf.fit(x_train, y_train)
x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
y_pred = clf.predict(x_eval)
np.savetxt("output/p03d_pred_SGD.txt", y_pred)
plt.figure()
plt.plot(y_eval, y_pred, 'bx')
plt.xlabel('true counts')
plt.ylabel('predict counts')
plt.savefig('output/p03d_SGD.png')