import numpy as np
import matplotlib.pyplot as plt
from functions import sigmoid, square, leaky_relu, chain_length_2, chain_deriv_2, chain_length_3, chain_deriv_3

def plot_chain(chain, file_name):
    input = np.arange(-3, 3, 0.01)
    if len(chain) == 2:
        output = chain_length_2(chain, input)
        derivative = chain_deriv_2(chain, input)
    elif len(chain) == 3:
        output = chain_length_3(chain, input)
        derivative = chain_deriv_3(chain, input)
    plt.plot(input, output, label="f(x)")
    plt.plot(input, derivative, label="df(x)/dx")
    plt.legend()
    plt.savefig(file_name)
    plt.clf()

plot_chain([square, sigmoid], 'figure1.png')
plot_chain([sigmoid, square], 'figure2.png')
plot_chain([leaky_relu, sigmoid, square], 'figure3.png')

