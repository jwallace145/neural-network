import math
import numpy as np
import matplotlib.pyplot as plt

from neural_network import NeuralNetwork
from activation_functions import sigmoid
from activation_functions import tanh
from activation_functions import relu


def main():
    # create a neural network
    neural_network = NeuralNetwork(1, 1, sigmoid)

    # generate data set
    x = np.linspace(-2*math.pi, 2*math.pi, 200)[:, None]
    y = np.cos(x)

    neural_network.learn(x, y, 10000)

    # plt.plot(x, y)
    # plt.plot(x, neural_network.run(x))
    # plt.show()
    plt.scatter(x, y)
    plt.scatter(x, neural_network.run(x))
    plt.show()






if __name__ == '__main__':
    main()
