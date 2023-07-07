import sys
import os

sys.path.append(r'C:\Users\Bos\Documents\Programmation\ML\ML_Library')

import math
import numpy as np

from Utils.preprocessing import polynomiale_features, MinMax_Scaler, Standard_Scaler
from Utils.metrics import R2_score
import matplotlib.pyplot as plt

print("Connected")



class Regression(object):
    """ Model de Régression élémentaire """

    def __init__(self, n_iterations, learning_rate):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.training_errors = []

    def init_weights(self, n_features):
        """ Initialise les différents poids de manière aléatoire suivant une loi uniforme """
        limit = 1 / math.sqrt(n_features)
        self.W = np.random.uniform(-limit, limit, (n_features, ))
    
    def fit(self, X, y):

        X = np.insert(X, 0, 1, axis=1)
        self.init_weights(n_features = X.shape[1])

        for i in range(self.n_iterations):
            y_pred = X.dot(self.W)

            mse = (1/2) * np.mean((y_pred - y)**2) + self.regularization(self.W)
            self.training_errors.append(mse)

            grad_W = 1/len(y) * (X.T.dot(y_pred - y))+ self.regularization.gradient(self.W)
            
            self.W -= self.learning_rate * grad_W


    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.W)
        return y_pred
    
    def get_score(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.W)
        return R2_score(y, y_pred)


class LinearRegression(Regression):

    def __init__(self, n_iterations = 100, learning_rate = 0.01, gradient_descent = True):

        self.gradient_descent = gradient_descent

        self.regularization = lambda x: 0
        self.regularization.gradient = lambda x: 0
        super(LinearRegression, self).__init__(n_iterations= n_iterations, learning_rate=learning_rate)

    
    def fit(self, X, y):

        print("Fitting the Linear Regression model on the given dataset ...")

        if not self.gradient_descent:
            X = np.insert(X, 0, 1, axis=1)
            # Calculate weights by least squares (using Moore-Penrose pseudoinverse)
            U, S, V = np.linalg.svd(X.T.dot(X))
            S = np.diag(S)
            X_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
            self.W = X_sq_reg_inv.dot(X.T).dot(y)
        else:
            super(LinearRegression, self).fit(X, y)




class PolynomyaleRegression(Regression):

    def __init__(self, degree, n_iterations = 100, learning_rate = 0.01):
        self.degree = degree
        self.regularization = lambda x: 0
        self.regularization.gradient = lambda x: 0
        super(PolynomyaleRegression, self).__init__(n_iterations= n_iterations, learning_rate= learning_rate)
    
    def fit(self, X, y):
        print("Fitting the Polynomiale Regression model on the given dataset ...")
        X = polynomiale_features(X, self.degree)
        super(PolynomyaleRegression, self).fit(X, y)


    def predict(self, X):
        X_new = polynomiale_features(X, self.degree)
        return super().predict(X_new)
    
    def get_score(self, X, y):
        X_new = polynomiale_features(X, self.degree)
        return super().get_score(X_new, y)




######## REGULARIZED REGRESSIONS ########




class l1_regularization():
    """ Regularization for Lasso Regression """
    def __init__(self, alpha_reg):
        self.alpha = alpha_reg
    
    def __call__(self, w):
        return self.alpha * np.linalg.norm(w)

    def gradient(self, w):
        return self.alpha * np.sign(w)



class l2_regularization():
    """ Regularization for Ridge Regression """
    def __init__(self, alpha_reg):
        self.alpha = alpha_reg
    
    def __call__(self, w):
        return self.alpha * 0.5 *  w.T.dot(w)

    def gradient(self, w):
        return self.alpha * w

class l1_l2_regularization():
    """ Regularization for Elastic Net Regression """
    def __init__(self, alpha_reg, l1_ratio=0.5):
        self.alpha = alpha_reg
        self.l1_ratio = l1_ratio

    def __call__(self, w):
        l1_contr = self.l1_ratio * np.linalg.norm(w)
        l2_contr = (1 - self.l1_ratio) * 0.5 * w.T.dot(w) 
        return self.alpha * (l1_contr + l2_contr)

    def gradient(self, w):
        l1_contr = self.l1_ratio * np.sign(w)
        l2_contr = (1 - self.l1_ratio) * w
        return self.alpha * (l1_contr + l2_contr) 



class LassoRegression(Regression):
    def __init__(self, degree, alpha_reg, n_iterations = 100, learning_rate = 0.01):

        self.degree = degree
        self.regularization = l1_regularization(alpha_reg)
        super(LassoRegression, self).__init__(n_iterations= n_iterations, learning_rate= learning_rate)
    
    def fit(self, X, y):
        print("Fitting the Lasso Regression model on the given dataset ...")
        self.scaler = Standard_Scaler()

        X_poly = polynomiale_features(X, self.degree)
        self.scaler.fit(X_poly)
        X = self.scaler.transform(X_poly) # Scale par min max
        super(LassoRegression, self).fit(X, y)

    def predict(self, X):
        X_poly = polynomiale_features(X, self.degree)
        X = self.scaler.transform(X_poly) # Re use the scaler alreary use and fit from the train_set
        return super(LassoRegression, self).predict(X)

    def get_score(self, X, y):
        X_poly = polynomiale_features(X, self.degree)
        X_new = self.scaler.transform(X_poly)
        return super().get_score(X_new, y)



class RidgeRegression(Regression):
    def __init__(self, degree, alpha_reg, n_iterations = 100, learning_rate = 0.01):

        self.degree = degree
        self.regularization = l2_regularization(alpha_reg)
        super(RidgeRegression, self).__init__(n_iterations= n_iterations, learning_rate= learning_rate)
    
    def fit(self, X, y):
        print("Fitting the Ridge Regression model on the given dataset ...")
        self.scaler = Standard_Scaler()

        X_poly = polynomiale_features(X, self.degree)
        self.scaler.fit(X_poly)
        X = self.scaler.transform(X_poly) # Scale par min max
        super(RidgeRegression, self).fit(X, y)

    def predict(self, X):
        X_poly = polynomiale_features(X, self.degree)
        X = self.scaler.transform(X_poly) # Re use the scaler alreary use and fit from the train_set
        return super(RidgeRegression, self).predict(X)

    def get_score(self, X, y):
        X_poly = polynomiale_features(X, self.degree)
        X_new = self.scaler.transform(X_poly)
        return super().get_score(X_new, y)



class ElasticNet(Regression):

    def __init__(self, degree, alpha_reg, n_iterations = 100, learning_rate = 0.01):

        self.degree = degree
        self.regularization = l1_l2_regularization(alpha_reg)
        super(ElasticNet, self).__init__(n_iterations= n_iterations, learning_rate= learning_rate)
    
    def fit(self, X, y):
        print("Fitting the ElasticNet model on the given dataset ...")
        self.scaler = Standard_Scaler()

        X_poly = polynomiale_features(X, self.degree)
        self.scaler.fit(X_poly)
        X = self.scaler.transform(X_poly) # Scale par min max
        super(ElasticNet, self).fit(X, y)

    def predict(self, X):
        X_poly = polynomiale_features(X, self.degree)
        X = self.scaler.transform(X_poly) # Re use the scaler alreary use and fit from the train_set
        return super(ElasticNet, self).predict(X)

    def get_score(self, X, y):
        X_poly = polynomiale_features(X, self.degree)
        X_new = self.scaler.transform(X_poly)
        return super(ElasticNet, self).get_score(X_new, y)






######## SGDR Regressor ########

# A faire #
