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
                  input_range: ndarray) -> ndarray:
    assert len(chain) == 2, "Chain must have length 2"
    assert input_range.ndim == 1, "Input range must be 1D"
    f1 = chain[0]
    f2 = chain[1]
    df1dx = deriv(f1, input_range)
    df2du = deriv(f2, f1(input_range))
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

