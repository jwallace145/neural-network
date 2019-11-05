import numpy as np

def tanh(x, derivative=False):
    if derivative:
        return 1 - np.tanh(x)**2
    return np.tanh(x)


def sigmoid(x, derivative=False):
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))

def relu(x, derivative=False):
    if derivative:
        x[x <= 0] = 0
        x[x > 0] = 1
        return x
    return np.maximum(0, x)

def leaky_relu(x, leaky_slope, derivative=False):
    if derivative:
        return
    return np.maximum(leaky_slope * x, x)

