import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    model_c = LogisticRegression()
    model_c.fit(x_train, t_train)
    t_pred = model_c.predict(x_test)
    np.savetxt(pred_path_c, t_pred > 0.5, fmt="%d")
    util.plot(x_test, t_test, model_c.theta, f"output/p02c_pred")
    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col='y', add_intercept=True)
    model_d = LogisticRegression()
    model_d.fit(x_train, y_train)
    y_pred = model_d.predict(x_test)
    np.savetxt(pred_path_d, y_pred > 0.5, fmt="%d")
    util.plot(x_test, y_test, model_d.theta, f"output/p02d_pred")
    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    x_valid, y_valid = util.load_dataset(valid_path, label_col='y', add_intercept=True)
    alpha = np.mean(model_d.predict(x_valid))
    t_pred_with_y = y_pred / alpha
    np.savetxt(pred_path_e, t_pred_with_y > 0.5, fmt="%d")
    # line: exp(-theta'^T x) + 1 = 2 / alpha
    # theta'^T x + log(2 / alpha - 1) = 0
    # To take x1 and x2 as variable, what we can change is the theta[0]
    # theta'[0] = theta[0] + log(2 / alpha - 1) = correction * theta[0]
    # correction = 1 + np.log(2 / alpha - 1) / theta[0]
    correction = 1 + np.log(2 / alpha - 1) / model_d.theta[0]
    util.plot(x_test, t_test, model_d.theta, f"output/p02e_pred", correction = correction)
    # *** END CODER HERE
