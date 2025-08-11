import numpy as np
from numpy import ndarray
from abc import ABC, abstractmethod

class Operation(ABC):
    @abstractmethod
    def forward(self, X: ndarray) -> ndarray:
        pass
    @abstractmethod
    def backward(self, output_grad: ndarray) -> ndarray:
        pass

class WeightMultiply(Operation):
    def __init__(self, W: ndarray):
       self.param = W
    def forward(self, X: ndarray):
       return np.dot(X, self.param)
    def backward(self, output_grad: ndarray) -> ndarray:
       self.param_grad = np.dot(output_grad, self.param.T)
       return self.param_grad

class BiasAdd(Operation):
    def __init__(self, B: ndarray):
        self.param = B
    def forward(self, X: ndarray) -> ndarray:
        return X + self.param
    def backward(self, output_grad: ndarray) -> ndarray:
        self.param_grad = output_grad.sum(axis=0)
        return self.param_grad

class Linear(Operation):
    def forward(self, X: ndarray) -> ndarray:
        return X
    def backward(self, output_grad: ndarray) -> ndarray:
        return output_grad

class Sigmoid(Operation):
    def forward(self, X: ndarray) -> ndarray:
        self.output = 1 / (1 + np.exp(-X))
        return self.output
    def backward(self, output_grad: ndarray) -> ndarray:
        sigmoid_derivative = self.output * (1 - self.output)
        return output_grad * sigmoid_derivative    
