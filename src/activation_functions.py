import math


def sigmoid(x):
    return 1.0/(1.0 + math.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def ReLU(x):
    return x if x >= 0.0 else 0.0


def ReLU_derivative(x):
    return 1.0 if x >= 0 else 0.0



FUNCTIONS = {
    'sigmoid': {
        'function': sigmoid,
        'derivative': sigmoid_derivative
    },
    'ReLU':{
        'function': ReLU,
        'derivative': ReLU_derivative
    }
}