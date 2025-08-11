from numpy import ndarray
from abc import ABC, abstractmethod

class Loss(ABC):
    @abstractmethod
    def forward(self, y_true: ndarray, y_pred: ndarray) -> float:
        pass
    @abstractmethod
    def backward(self, y_true: ndarray, y_pred: ndarray) -> ndarray:
        pass

class MeanSquaredError(Loss):
    def forward(self, y_true: ndarray, y_pred: ndarray) -> float:
        return ((y_true - y_pred) ** 2).mean()
    def backward(self, y_true: ndarray, y_pred: ndarray) -> ndarray:
        return -2 * (y_true - y_pred) / y_true.size

