import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

class KNNClassifier:
    def __init__(self, n_neighbors=5, weights='uniform'):
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights
        )
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

class KNNRegressor:
    def __init__(self, n_neighbors=5, weights='uniform'):
        self.model = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights
        )
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X) 