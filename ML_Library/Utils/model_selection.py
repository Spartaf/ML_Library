import numpy as np

def shuffle_data(X, y, seed = None):
    if seed:
        np.random.seed(seed)
    rand_list = np.arange(X.shape[0])
    np.random.shuffle(rand_list)
    return X[rand_list], y[rand_list]


def train_test_split(X, y, test_size = 0.2, shuffle = True, seed = None):
    if shuffle:
        X, y = shuffle_data(X, y, seed=seed)

    n_sample = X.shape[0]
    i_max = int(n_sample * test_size)
    X_train, X_test = X[i_max:], X[:i_max]
    y_train, y_test = y[i_max:], y[:i_max]

    return X_train, y_train, X_test, y_test


def divide_on_feature(X, feature_i, threshold):
    """ Divide dataset based on if sample value on feature index is larger than
        the given threshold """
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        # cas quantitatif
        split_func = lambda sample: sample[feature_i] >= threshold
    else:
        # cas qualitatif
        split_func = lambda sample: sample[feature_i] == threshold

    X_1 = np.array([sample for sample in X if split_func(sample)])
    X_2 = np.array([sample for sample in X if not split_func(sample)])

    return np.array([X_1, X_2])


    

