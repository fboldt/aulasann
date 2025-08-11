from abc import ABC, abstractmethod
from numpy import ndarray
from Operation import Operation, WeightMultiply, BiasAdd, Linear
import numpy as np

class Layer(ABC):
    def __init__(self):
        self.operations = []
        self.params = []
        self.params_grad = []
        self.initiazed = False

    @abstractmethod
    def _initialize(self, input_shape: tuple):
        pass
    
    def forward(self, X: ndarray) -> ndarray:
        if not self.initiazed:
            self._initialize(X.shape)
        self.input_ = X
        for operation in self.operations:
            X = operation.forward(X)
        self.output_ = X
        return self.output_

    def backward(self, output_grad: ndarray) -> ndarray:
        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)
        input_grad = output_grad
        return input_grad


class Dense(Layer):
    def __init__(self, neurons: int, activation: Operation = Linear()):
        self.params = []
        self.neurons = neurons
        self.activation = activation
        self.learning_rate = 1

    def _initialize(self, input_shape: tuple):
        self.operations = [
            WeightMultiply(np.random.randn(input_shape[1], self.neurons) * 0.01),
            BiasAdd(np.zeros((1, self.neurons))),
            self.activation
        ]
        self.initiazed = True   
