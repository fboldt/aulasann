import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score


def criaDataset(n=20, slop=[2,1], intercept=-0.4):
  X = np.random.uniform(size=(n,2))
  AUX = X * np.array(slop) - [0, intercept]
  y = np.array(AUX[:,0]>AUX[:,1], dtype=int)*2-1
  return X, y

X, y = criaDataset()

sign = lambda a: (a>=0)*2-1
include_bias = lambda X: np.c_[np.ones(X.shape[0]), X]

class Adaline(BaseEstimator, ClassifierMixin):
  def __init__(self, epochs=1000, learning_rate=0.01):
    self.epochs = range(epochs)
    self.learning_rate = learning_rate

  def fit(self, X, y):
    Xb = include_bias(X)
    self.w = np.random.uniform(size=Xb.shape[1])*2-1
    for _ in self.epochs:
        ypred = (Xb @ self.w)
        error = y-ypred
        self.w += Xb.T @ error * self.learning_rate
        cost = sum(error**2)
        if cost == 0:
          break
    return self

  def predict(self, X):
    Xb = include_bias(X)
    return sign(Xb @ self.w)

model = Adaline()
model.fit(X, y)
ypred = model.predict(X)
print(accuracy_score(y, ypred))
