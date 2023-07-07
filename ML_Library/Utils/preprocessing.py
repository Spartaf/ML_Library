from itertools import combinations_with_replacement
import numpy as np 
import math


class MinMax_Scaler():
    def __init__(self, feature_range=(0, 1)):
        self.low, self.high = feature_range
        self.Xmin = 0
        self.Xmax = 0

    def fit(self, X):
        self.Xmin = np.min(X,axis=0)
        self.Xmax = np.max(X,axis=0)

    def transform(self, X):
        X_std = (X - self.Xmin) / (self.Xmax - self.Xmin)
        return X_std * (self.high - self.low) + self.low

    def fit_transform(self, X):
        return self.fit(X).transform(X)



class Standard_Scaler():
    def __init__(self):
        self.Xmean = 0
        self.Xstd = 0

    def fit(self, X):
        self.Xmean = X.mean(axis=0)
        self.Xstd = X.std(axis=0)

    def transform(self, X):
        X_copy = X
        for col in range(X.shape[1]):
            if self.Xstd[col]:
                X_copy[:, col] = (X_copy[:, col] - self.Xmean[col]) / self.Xstd[col]
        return X_copy
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)



def polynomiale_features(X, degree = 2):
    """ Si X = [A,B] et degree = 2 ==> new_X = [1, A, B, A2, AB, B2] on a AB(intéraction)  """

    n_samples, n_features = np.shape(X)

    def index_combinations():
        combs = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
        # combinations_with_replacement([A, B, C], 2) = AA AB AC BB BC CC (ordre lexicographique)
        flat_combs = [item for sublist in combs for item in sublist]
        # on récupère toutes les différentes combinaisons dans une liste
        return flat_combs
    
    combinaisons = index_combinations()
    n_output_features = len(combinaisons) # Autant de colonnes que de combinaisons
    # X = [A, B] et degree = 2 ==> flat_combs = [(), (A,), (B,), (A,A), (A,B), (B,B)]
    #                          ==> new_X = [1, A, B, A2, AB, B2] donc même taille

    new_X = np.empty((n_samples, n_output_features))

    for i, index_comb in enumerate(combinaisons):
        new_X[:, i] = np.prod(X[:, index_comb], axis = 1) # axis 1 pour ligne

    return new_X 


    

