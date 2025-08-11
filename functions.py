import numpy as np
from numpy import ndarray
from typing import Callable, List

def square(x: ndarray) -> ndarray:
    return np.power(x, 2)

def leaky_relu(x: ndarray) -> ndarray:
    return np.maximum(0.2 * x, x)

def deriv(func: Callable[[ndarray], ndarray],
          input_: ndarray,
          delta: float = 1e-5) -> ndarray:
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)

Array_Function = Callable[[ndarray], ndarray]

Chain = List[Array_Function]

def chain_length_2(chain: Chain,
                   a: ndarray) -> ndarray:
    assert len(chain) == 2, "Chain must have length 2"
    f1 = chain[0]
    f2 = chain[1]
    return f2(f1(a))

def sigmoid(x: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-x))

def chain_deriv_2(chain: Chain,
                  input: ndarray) -> ndarray:
    assert len(chain) == 2, "Chain must have length 2"
    assert input.ndim == 1, "Input range must be 1D"
    f1 = chain[0]
    f2 = chain[1]
    df1dx = deriv(f1, input)
    df2du = deriv(f2, f1(input))
    return df1dx * df2du

def chain_length_3(chain: Chain,
                   a: ndarray) -> ndarray:
    assert len(chain) == 3, "Chain must have length 3"
    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]
    return f3(f2(f1(a)))

def chain_deriv_3(chain: Chain,
                  input: ndarray) -> ndarray:
    assert len(chain) == 3, "Chain must have length 3"
    assert input.ndim == 1, "Input range must be 1D"
    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]
    df1dx = deriv(f1, input)
    f1_of_x = f1(input)
    df2du = deriv(f2, f1_of_x)
    f2_of_f1 = f2(f1_of_x)
    df3dv = deriv(f3, f2_of_f1)
    return df1dx * df2du * df3dv

def multiple_inputs_add(x: ndarray,
                        y: ndarray,
                        sigma: Array_Function) -> float:
    assert x.shape == y.shape
    a = x + y
    return sigma(a)

def multiple_inputs_add_backward(x: ndarray,
                                 y: ndarray,
                                 sigma: Array_Function) -> float:
    a = x + y
    dsda = deriv(sigma, a)
    dadx, dady = 1, 1
    return dsda * dadx, dsda * dady

def matmul_forward(X: ndarray,
                   W: ndarray) -> ndarray:
    assert X.shape[1] == W.shape[0]
    N = np.dot(X, W)
    return N

def matmul_backward_first(X: ndarray,
                          W: ndarray) -> ndarray:
    dNdX = np.transpose(W, (1, 0))
    return dNdX

def matrix_forward_extra(X: ndarray,
                         W: ndarray,
                         sigma: Array_Function) -> ndarray:
    assert X.shape[1] == W.shape[0]
    N = np.dot(X, W)
    S = sigma(N)
    return S

def matrix_function_backward_1(X: ndarray,
                               W: ndarray,
                               sigma: Array_Function) -> ndarray:
    assert X.shape[1] == W.shape[0]
    N = np.dot(X, W)
    dSdN = deriv(sigma, N)
    dNdX = np.transpose(W, (1, 0))
    return np.dot(dSdN, dNdX)

def matrix_function_forward_sum(X: ndarray,
                                W: ndarray,
                                sigma: Array_Function) -> float:
    assert X.shape[1] == W.shape[0]
    N = np.dot(X, W)
    S = sigma(N)
    L = np.sum(S)
    return L

def matrix_function_backward_sum_1(X: ndarray,
                                   W: ndarray,
                                   sigma: Array_Function) -> ndarray:
    assert X.shape[1] == W.shape[0]
    N = np.dot(X, W)
    dSdN = deriv(sigma, N)
    dNdX = np.transpose(W, (1, 0))
    dLdX = np.dot(dSdN, dNdX)
    return dLdX

