import numpy as np 


def make_diagonal(x):
    """ Converts a vector into an diagonal matrix """
    m = np.zeros((len(x), len(x)))
    for i in range(m.shape[0]):
        m[i, i] = x[i]
    return m