import numpy as np
from sklearn.naive_bayes import GaussianNB

class NaiveBayes:
    def __init__(self):
        self.model = GaussianNB()
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X) 