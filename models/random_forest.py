import numpy as np
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

class RandomForestClassifier:
    def __init__(self):
        self.model = SklearnRandomForestClassifier(n_estimators=100)
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

class CustomRandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=None):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth
        )
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X) 