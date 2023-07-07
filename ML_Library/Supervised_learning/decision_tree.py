

# PAS FINI 

import sys
import os

sys.path.append(r'C:\Users\Bos\Documents\Programmation\ML\ML_Library')

import math
import numpy as np
import matplotlib.pyplot as plt

from ML_Library.Utils import divide_on_feature
from ML_Library.Utils import mean_squared_error


class Decision_Node():
    
    def __init__(self, feature_i = None, threshold = None, isleaf = None, true_branch = None, false_branch = None):
        self.feature_i = feature_i
        self.threshold = threshold
        self.isleaf = isleaf
        self.true_branch = true_branch
        self.false_branch = false_branch


class Decision_Tree():

    def __init__(self, min_samples = 2, min_impurity = 1e-7, max_depth = float("inf"), loss = None):

        # Nombre minimum de données pour chaque split
        self.min_samples = min_samples
        # Impuretée minimum à ne pas dépasser
        self.min_impurity = min_impurity
        # Profondeur maximum de l'arbre de décision
        self.max_depth = max_depth
        # Fonction de cout utilisé en cas de gradient boosting
        self.loss = loss

    def create_tree(self, X, y, current_depth = 0):
        """ Fonction récursive de création de l'arbre """

        best_impurity_gain = 0
        best_critere = None
        best_split = None

        if len(y.shape) == 1:
            y = np.expand_dims(y, axis = 1)

        Xy = np.concatenate((X, y), axis=1)

        n_samples, n_features = np.shape(X)

        if n_samples >= self.min_samples and current_depth <= self.max_depth:

            # parcours de chaques features
            for feature_i in range(n_features):

                feature_values = np.expand_dims(X[:, feature_i], axis = 1)
                unique_values = np.unique(feature_values)

                # parcours de chaque valeurs unique possible pour le seuil
                for threshold in unique_values:

                    # Sépartion de la feature_i (colonne) en fonction du seuil
                    Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)

                    # Calcule de l'impurté
                    if len(Xy1) > 0 and len(Xy2) > 0:
                        # Récupération des y dans les deux split
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]

                    # Entropy, Gini
                    impurity_gain = self._impurity_calculation(y, y1, y2)

                    if impurity_gain > best_impurity_gain:
                        # Récupération du split optimum
                        best_impurity = impurity
                        best_critere = {"feature_i": feature_i, "threshold": threshold}
                        best_split = {
                                "leftX": Xy1[:, :n_features],   # X of left subtree
                                "lefty": Xy1[:, n_features:],   # y of left subtree
                                "rightX": Xy2[:, :n_features],  # X of right subtree
                                "righty": Xy2[:, n_features:]   # y of right subtree
                                }
            if largest_impurity > self.min_impurity:
            # Build subtrees for the right and left branches
                true_branch = self.create_tree(best_split["leftX"], best_split["lefty"], current_depth + 1)

                false_branch = self.create_tree(best_split["rightX"], best_split["righty"], current_depth + 1)

                return Decision_Node(feature_i=best_critere["feature_i"], threshold=best_critere[
                                    "threshold"], true_branch=true_branch, false_branch=false_branch)
            

            leaf_value = self._leaf_value_calculation(y)

            return DecisionNode(value=leaf_value)
                                







