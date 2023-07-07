import math
import numpy as np
import math



def mean_squared_error(y_true, y_pred):
    """ Returns the mean squared error between y_true and y_pred """
    mse = (1/2) * np.mean(np.power(y_true - y_pred, 2))
    return mse


def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy



def variance_features(X):
    """ Return variance features """
    n = X.shape[0]
    mean = np.ones(X.shape) * X.mean(0)
    variance = (1 / n_samples) * np.diag((X - mean).T.dot(X - mean))
    return variance


def R2_score(y_true, y_pred):
    rss = np.sum((y_pred - y_true)**2)
    tss = np.sum((y_true - np.mean(y_true))**2)

    r2_score = 1 - (rss/tss)
    return r2_score


def euclidean_distance(a, b):
    """ euclidiea distance between two vectors """
    return np.sqrt(np.sum(np.power(a - b, 2)))

def calculate_entropy(y):
    """ Calculate the entropy of label array y """
    log2 = lambda x: math.log(x) / math.log(2)
    unique_labels = np.unique(y)
    entropy = 0
    for label in unique_labels:
        count = len(y[y == label])
        p = count / len(y)
        entropy += -p * log2(p)
    return entropy



