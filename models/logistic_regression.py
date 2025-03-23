import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

class LogisticRegression:
    def __init__(self):
        self.model = SklearnLogisticRegression(multi_class='ovr')
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X) 