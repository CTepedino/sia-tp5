import numpy as np

#iguales a las otras en escalares, pero tambien se pueden aplicar directamente a un array de np


def identity(x):
    return x

def identity_derivative(x):
    return np.ones(x.shape)

def hyperbolic_tangent(x):
    return np.tanh(x)

def hyperbolic_tangent_derivative(x):
    return 1 - np.power(np.tanh(x), 2)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s*(1-s)


def relu(x):
    return x * (x > 0)

def relu_derivative(x):
    return 1 * (x > 0)

activator_functions = {
    "identity": (identity, identity_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (hyperbolic_tangent, hyperbolic_tangent_derivative),
    "relu": (relu, relu_derivative),
}