import numpy as np
from sklearn.svm import SVC
from sklearn.svm import SVR

class SVMClassifier:
    def __init__(self, kernel='rbf', C=1.0):
        self.model = SVC(
            kernel=kernel,
            C=C,
            random_state=42
        )
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

class SVMRegressor:
    def __init__(self, kernel='rbf', C=1.0):
        self.model = SVR(
            kernel=kernel,
            C=C
        )
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X) 