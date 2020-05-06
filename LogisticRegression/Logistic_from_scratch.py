import numpy as np
from statsmodels.gam.tests.test_gam import sigmoid


def compute_prediction(X, weights):
    """
    Compute prediction y_hat based on current weights
    :param X: numpy.ndarray
    :param weights: numpy.ndarray
    :return: numpy.ndarray, y_hat of X under weights
    """
    z = np.dot(X, weights)  # dot product
    predictions = sigmoid(z)
    return predictions


def update_weights(x_train, y_train, weights, learning_rate):
    """
    Update weights by one step
    :param x_train: numpy.ndarray, training data set
    :param y_train: numpy.ndarray, training data set
    :param weights: numpy.ndarray
    :param learning_rate: float
    :return: numpy.ndarray, updated weights
    """
    predictions = compute_prediction(x_train, weights)
    weights_delta = np.dot(x_train.T, y_train - predictions)
    m = y_train.shape[0]
    weights += learning_rate / float(m) * weights_delta
    return weights


def calculate_cost(x, y, weights):
    """
    Compute the cost J(w)
    :param x: numpy.ndarray, training data set
    :param y: numpy.ndarray, training data set
    :param weights: numpy.ndarray
    :return: float
    """
    predictions = compute_prediction(x, weights)
    cost = np.mean(-y * np.log(predictions) - (1 - y) * np.log(1 - predictions))
    return cost


def train_logistic_regression(x_train, y_train, learning_rate, fit_intercept=False, max_iter=500):
    """
    Train a logistic regression model
    :param x_train: numpy.ndarray, training data set
    :param y_train: numpy.ndarray, training data set
    :param max_iter: int, number of max iterations
    :param learning_rate: float
    :param fit_intercept: bool with an intercept w0 or not
    :return: numpy.ndaaray, learned weights
    """
    if fit_intercept:
        intercept = np.ones(x_train.shape[0], 1)
        x_train = np.hstack((intercept, x_train))  # hstacks merges 2 arrays column wise
    weights = np.zeros(x_train.shape[1])
    for iteration in range(max_iter):
        weights = update_weights(x_train, y_train, weights, learning_rate)
        # printing cost for every 100 iterations
        if iteration % 100 == 0:
            print(calculate_cost(x_train, y_train, weights))
    return weights


def predict(x, weights):
    if x.shape[1] == weights.shape[0] - 1:
        intercept = np.ones((x.shape[0], 1))
        x = np.hstack((intercept, x))
    return compute_prediction(x, weights)
