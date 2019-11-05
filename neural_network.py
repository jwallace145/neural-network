# important necessary modules
import numpy as np


# neural network class
class NeuralNetwork(object):

    def __init__(self, input_dim, output_dim, activation_function, hidden_nodes=10, learning_rate=0.001):
        # seed the random module
        np.random.seed(1)

        # set the activation function
        self.activation_function = activation_function

        # set the learning rate of the neural network
        self.learning_rate = learning_rate

        # create the weight matrices of the three hidden layers
        self.w1 = np.random.uniform(0, 1, (input_dim, hidden_nodes))
        self.w2 = np.random.uniform(0, 1, (hidden_nodes, hidden_nodes))
        self.w3 = np.random.uniform(0, 1, (hidden_nodes, output_dim))

        # create biases
        self.b1 = np.full((1, hidden_nodes), 0.1)
        self.b2 = np.full((1, hidden_nodes), 0.1)
        self.b3 = np.full((1, 1), 0.1)

    def tanh(self, x):
        return np.tanh(x)

    def derivative_tanh(self, x):
        return 1 - np.tanh(x) ** 2

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def run(self, x):
        # tanh activation function
        # a1 = x
        # z2 = a1.dot(self.w1) + self.b1
        # a2 = self.tanh(z2)
        # z3 = a2.dot(self.w2) + self.b2
        # a3 = self.tanh(z3)
        # z4 = a3.dot(self.w3) + self.b3
        # return z4

        # sigmoid activation function
        # a1 = x
        # z2 = a1.dot(self.w1) + self.b1
        # a2 = self.sigmoid(z2)
        # z3 = a2.dot(self.w2) + self.b2
        # a3 = self.sigmoid(z3)
        # z4 = a3.dot(self.w3) + self.b3
        # return z4

        a1 = x
        z2 = a1.dot(self.w1) + self.b1
        a2 = self.activation_function(z2)
        z3 = a2.dot(self.w2) + self.b2
        a3 = self.activation_function(z3)
        z4 = a3.dot(self.w3) + self.b3
        return z4

    def learn(self, x, y, iterations):
        # tanh activation function
        # for i in range(iterations):
        #     a1 = x
        #     z2 = a1.dot(self.w1) + self.b1
        #     a2 = self.tanh(z2)
        #     z3 = a2.dot(self.w2) + self.b2
        #     a3 = self.tanh(z3)
        #     z4 = a3.dot(self.w3) + self.b3
        #
        #     cost = np.sum((z4 - y)**2)/2
        #
        #     # backpropagation
        #     z4_delta = z4 - y
        #     dw3 = a3.T.dot(z4_delta)
        #     db3 = np.sum(z4_delta, axis=0, keepdims=True)
        #
        #     z3_delta = z4_delta.dot(self.w3.T) * self.derivative_tanh(z3)
        #     dw2 = a2.T.dot(z3_delta)
        #     db2 = np.sum(z3_delta, axis=0, keepdims=True)
        #
        #     z2_delta = z3_delta.dot(self.w2.T) * self.derivative_tanh(z2)
        #     dw1 = x.T.dot(z2_delta)
        #     db1 = np.sum(z2_delta, axis=0, keepdims=True)
        #
        #     # update parameters
        #     for param, gradient in zip([self.w1, self.w2, self.w3, self.b1, self.b2, self.b3], [dw1, dw2, dw3, db1, db2, db3]):
        #         param -= self.learning_rate * gradient

        # sigmoid activation function
        # for i in range(iterations):
        #     a1 = x
        #     z2 = a1.dot(self.w1) + self.b1
        #     a2 = self.sigmoid(z2)
        #     z3 = a2.dot(self.w2) + self.b2
        #     a3 = self.sigmoid(z3)
        #     z4 = a3.dot(self.w3) + self.b3
        #
        #     cost = np.sum((z4 - y)**2)/2
        #
        #     # backpropagation
        #     z4_delta = z4 - y
        #     dw3 = a3.T.dot(z4_delta)
        #     db3 = np.sum(z4_delta, axis=0, keepdims=True)
        #
        #     z3_delta = z4_delta.dot(self.w3.T) * self.sigmoid_derivative(z3)
        #     dw2 = a2.T.dot(z3_delta)
        #     db2 = np.sum(z3_delta, axis=0, keepdims=True)
        #
        #     z2_delta = z3_delta.dot(self.w2.T) * self.sigmoid_derivative(z2)
        #     dw1 = x.T.dot(z2_delta)
        #     db1 = np.sum(z2_delta, axis=0, keepdims=True)
        #
        #     # update parameters
        #     for param, gradient in zip([self.w1, self.w2, self.w3, self.b1, self.b2, self.b3], [dw1, dw2, dw3, db1, db2, db3]):
        #         param -= self.learning_rate * gradient

        for i in range(iterations):
            a1 = x
            z2 = a1.dot(self.w1) + self.b1
            a2 = self.activation_function(z2)
            z3 = a2.dot(self.w2) + self.b2
            a3 = self.activation_function(z3)
            z4 = a3.dot(self.w3) + self.b3

            cost = np.sum((z4 - y)**2)/2

            # backpropagation
            z4_delta = z4 - y
            dw3 = a3.T.dot(z4_delta)
            db3 = np.sum(z4_delta, axis=0, keepdims=True)

            z3_delta = z4_delta.dot(self.w3.T) * self.activation_function(z3, True)
            dw2 = a2.T.dot(z3_delta)
            db2 = np.sum(z3_delta, axis=0, keepdims=True)

            z2_delta = z3_delta.dot(self.w2.T) * self.activation_function(z2, True)
            dw1 = x.T.dot(z2_delta)
            db1 = np.sum(z2_delta, axis=0, keepdims=True)

            # update parameters
            for param, gradient in zip([self.w1, self.w2, self.w3, self.b1, self.b2, self.b3], [dw1, dw2, dw3, db1, db2, db3]):
                param -= self.learning_rate * gradient
