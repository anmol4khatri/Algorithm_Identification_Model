import numpy as np
from sklearn.linear_model import Ridge

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.model = Ridge(
            alpha=alpha,
            random_state=42
        )
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X) 