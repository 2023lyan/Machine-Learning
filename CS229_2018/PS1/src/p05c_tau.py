import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    # Fit a LWR model with the best tau value
    # Run on the test set to get the MSE value
    # Save predictions to pred_path
    # Plot data
    tau_min = tau_values[0]
    mse_min = 1e9
    for tau in tau_values:
        clf = LocallyWeightedLinearRegression(tau)
        clf.fit(x_train, y_train)
        x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
        y_pred = clf.predict(x_eval)
        mse = np.mean((y_eval - y_pred) ** 2)
        if(mse < mse_min):
            tau_min = tau
            mse_min = mse
        print(f"MSE = {mse} for tau = {tau}")
        plt.figure()
        plt.title(f'tau = {tau}')
        plt.plot(x_train, y_train, 'bx', linewidth = 2)
        plt.plot(x_eval, y_pred, 'ro', linewidth = 2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f'output/p05c_tau={tau}.png')
    print(f"When tau = {tau_min}, mse archives the minimun value {mse_min}.")
    clf = LocallyWeightedLinearRegression(tau_min)
    clf.fit(x_train, y_train)
    x_eval, y_eval = util.load_dataset(test_path, add_intercept=True)
    y_pred = clf.predict(x_eval)
    mse = np.mean((y_eval - y_pred) ** 2)
    print(f"In the test dataset, mse = {mse}")
    np.savetxt(pred_path, y_pred)
    # *** END CODE HERE ***
