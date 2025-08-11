from Loss import Loss, MeanSquaredError
from numpy import ndarray

class NeuralNetwork():
    def __init__(self, layers: list, loss: Loss = MeanSquaredError()):
        super().__init__()
        self.layers = layers
        self.loss = loss

    def forward(self, X: ndarray) -> ndarray:
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, y_true: ndarray, y_pred: ndarray) -> ndarray:
        output_grad = self.loss.backward(y_true, y_pred)
        for layer in reversed(self.layers):
            output_grad = layer.backward(output_grad)
        return output_grad

