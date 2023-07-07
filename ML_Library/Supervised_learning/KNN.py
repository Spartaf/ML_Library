import sys
import os

sys.path.append(r'C:\Users\Bos\Documents\Programmation\ML\ML_Library')


import numpy as np
from ML_Library.Utils.metrics import euclidean_distance

print("connected")

class knn(object):
    def __init__(self, k = 5, distance_metric = euclidean_distance):
        self.k = k
        self.distance = distance_metric
    
    def fit(self, X, y):
        self.train_target = y
        self.train_feature = X

    def vote(self, x):

        # Distances computation
        dists = np.array([self.distance(x, x_train) for x_train in self.train_feature])

        # Indices of the k_nearest neighbors
        k_neighbors_indices = np.argsort(dists)[:self.k]

        # Labels of the K_nearest neighbors 
        k_neighbors_labels = np.array([self.train_target[i] for i in k_neighbors_indices])          

        # Number of each label
        counts = np.bincount(k_neighbors_labels.astype('int'))

        return counts.argmax()

    def predict(self, X):
        y_pred = [self.vote(x) for x in X]
        return np.array(y_pred)

    def get_score(self, X, y): 
        score = np.sum(self.predict(X) == y) / len(y)
        return score

    
    