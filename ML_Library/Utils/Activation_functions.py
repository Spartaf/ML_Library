import numpy as np 
import math 


class Sigmoid():
    def __call__(self, X):
        return 1 / (1 + np.exp(-X))
    
    def gradient(self, X):
        return self.__call__(X) * (1 - self.__call__(X))

    
class ReLu():
    def __call__(self, X): # __call__ permet d'etre utiliser comme fonction
        # np.where(x >= 0, x, 0)
        return np.maximum(X, 0)

    def gradient(self, X):
        return np.where(x >= 0, 1, 0)


class Softmax():
    def __call__(self, X):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def gradient(self, X):
        f = self.__call__(x)
        return f * (1 - f)


class Tanh():
    def __call__(self, X):
        # e_x = np.exp(X) 
        # e_mx = np.exp(-X)
        # return (e_x - e_mx) / (e_x + e_mx)
        return 2 / (1 + np.exp(-2*x)) - 1 # Pour moins de calculs

    def gradient(self, X):
        return 1 - np.power(self.__call__(x), 2)



