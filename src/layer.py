from matrix import Matrix
import random

class Layer():
    def __init__(self, activation: dict[str, callable], input_size: int, neurons: int):
        self.activation = activation.get('function')
        self.derivative = activation.get('derivative')

        self.neurons = neurons

        self.biases = Matrix(neurons, 1, lambda : random.uniform(-1.0, 1.0))
        self.weights = Matrix(neurons, input_size,  lambda : random.uniform(-1.0, 1.0))

    def forward(self, input: Matrix):
        self.input = input
        self.pre_activation_values = input.multiply(self.weights).add(self.biases)
        return self.pre_activation_values.apply(self.activation)

    
    def backward(self, output_gradients: Matrix, learning_rate):
        # dE/dZ * dZ/dA = dE/dA
        deda = output_gradients.multiply_element_wise(self.pre_activation_values.apply(self.derivative))

        input_gradient = deda.multiply(self.weights.transpose())

        weight_nudges = self.input.transpose().multiply(deda).multiply(-learning_rate)
        bias_nudges = deda.multiply(-learning_rate)

        self.weights = self.weights.add(weight_nudges)
        self.biases = self.biases.add(bias_nudges)

        return input_gradient