import sys
import os

sys.path.append(r'C:\Users\Bos\Documents\Programmation\ML\ML_Library')

import math
import numpy as np
import matplotlib.pyplot as plt

from Utils.Activation_functions import Sigmoid
from Utils import make_diagonal

print("Connected")

class Logistic_regression():

    def __init__(self, n_iterations = 1000, learning_rate=0.01, gradient_descent=True, multi_class = True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.gradient_descent = gradient_descent
        self.sigmoid = Sigmoid()
        self.training_errors = []
        self.multi_class = multi_class

    def init_weights(self, n_features):
        """ Initialise les différents poids de manière aléatoire suivant une loi uniforme """
        limit = 1 / math.sqrt(n_features)
        self.W = np.random.uniform(-limit, limit, (n_features, )) # on rajoute la colonne de 1 pour l'intersept apres

    def fit(self, X, y):
        print("Fitting the Logistic Regression model on the given dataset ...")
        X_copy = X
        X_copy = np.insert(X_copy, 0, 1, axis=1)
        n_samples, n1_features = X_copy.shape 
        self.init_weights(n_features = n1_features)


        # Multiclass Training with One-VS-All
        if self.multi_class:
            self.W = []
            self.training_errors = []

            for i in np.unique(y):
                y_onevsall = np.where(y == i, 1, 0)
                #self.init_weights(n_features= n1_features)
                cost = []
                theta = np.zeros(X_copy.shape[1])
            
                for _ in range(self.n_iterations):
                    y_pred = self.sigmoid(X_copy.dot(theta))
                    log_loss = (1 / n_samples) * (np.sum(-y_onevsall.T.dot(np.log(y_pred)) - (1 - y_onevsall).T.dot(np.log(1 - y_pred))))
                    cost.append(log_loss)
                    theta -= (self.learning_rate / n_samples) * np.dot(X_copy.T, (y_pred) - y_onevsall)

                self.W.append((theta, i))
                self.training_errors.append((cost, i))

        else:
            if self.gradient_descent:
                for i in range(self.n_iterations):
                    y_pred = self.sigmoid(X_copy.dot(self.W))
                
                    log_loss = 1/(2*n_samples) * np.dot((y - y_pred).T , (y - y_pred))
                    self.training_errors.append(log_loss)

                    self.W -= (self.learning_rate / n_samples) * np.dot(X_copy.T, (y_pred) - y)
                
            else:
                y_pred = self.sigmoid(X_copy.dot(self.W))
                # Make a diagonal matrix of the sigmoid gradient column vector
                diag_gradient = make_diagonal(self.sigmoid.gradient(X_copy.dot(self.W)))
                # Batch opt:
                self.W = np.linalg.pinv(X_copy.T.dot(diag_gradient).dot(X_copy)).dot(X_copy.T).dot(diag_gradient.dot(X_copy).dot(self.W) + y - y_pred)



    def predict(self, X):
        X_copy = X
        X_copy = np.hstack((np.ones((X.shape[0], 1)) , X_copy)) # On met dans la colonne de gauche que des 1 pour le biais
        if self.multi_class:
            y_predict = [max((self.sigmoid(i.dot(theta)), c) for theta, c in self.W)[1] for i in X_copy ]
            temp = []
            res = []
            for i in X_copy:
                for theta ,c in self.W:
                   temp.append((self.sigmoid(i.dot(theta)), c))

                res.append(max(temp)[1])
                temp = []

            return res
        else:
            return np.round(self.sigmoid(X_copy.dot(self.W))).astype(int)


    def get_score(self, X, y): 
        score = sum(self.predict(X) == y) / len(y)
        return score

    def fit_curve(self):

        if self.multi_class:
            plt.figure(figsize=(16,5))
            for cost,c in self.training_errors:
                plt.subplot(1,len(self.training_errors), int(c+1))
                plt.plot(range(len(cost)),cost,'r')
                plt.title(f"Fit Curve : category {c} VS All")
                plt.xlabel("Number of Iterations")
                plt.ylabel("Cost")
            plt.show()
        else:
            plt.figure(figsize=(10,5))
            plt.plot(self.training_errors , 'r')
            plt.title("Fit Curve")
            plt.xlabel("Number of Iterations")
            plt.ylabel("Cost")
            plt.show()

    def decision_boundary(self, X_train, y_train, X_test = None , y_test = None):
        plt.figure(figsize=(10,5))
        plt.subplot(1, 2, 1)
        x0, x1 = np.meshgrid(
        np.linspace(X_data[:,0].min(), X_data[:,0].max(), 500).reshape(-1, 1),
        np.linspace(X_data[:,1].min(), X_data[:,1].max(), 500).reshape(-1, 1)
        )

        X_new = np.c_[x0.ravel(), x1.ravel()]
        y_pred = np.array(model.predict(X_new))
        zz = y_pred.reshape(x0.shape)

        plt.figure(figsize=(8, 5))
        plt.contourf(x0, x1, zz)
        plt.scatter(X_data[:,0], X_data[:,1] , c=Y_data, edgecolor = 'k')
        plt.title('Scatter Plot with Decision Boundary for the Training Set')
        plt.tight_layout()
        plt.show()


    




