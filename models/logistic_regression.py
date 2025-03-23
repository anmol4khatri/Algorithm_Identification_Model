import numpy as np
from sklearn.linear_model import LogisticRegression

class LogisticRegression:
    def __init__(self, max_iter=1000):
        self.model = LogisticRegression(
            max_iter=max_iter,
            random_state=42
        )
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X) 