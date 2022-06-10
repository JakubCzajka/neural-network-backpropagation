from matrix import Matrix, Converter

from activation_functions import FUNCTIONS as ACTIVATION_FUNCTIONS
from loss_functions import FUNCTIONS as LOSS_FUNCTIONS
import pandas as pd

from network import NeuralNetwork


def xor_data():
    dataset = pd.read_csv('resources/xor.csv', delimiter=',')
    return Matrix.read_dataset(dataset)


def iris_data():
    dataset = pd.read_csv('resources/iris.data', delimiter=',')
    return *(Matrix.read_dataset(dataset)), Converter(dataset)


if __name__ == '__main__':
    
    x, y, converter = iris_data()

    network = NeuralNetwork()
    network.add_layer(3, ACTIVATION_FUNCTIONS['sigmoid'], x.columns)
    network.add_layer(7, ACTIVATION_FUNCTIONS['sigmoid'])
    network.add_layer(y.columns, ACTIVATION_FUNCTIONS['sigmoid'])

    network.train((x,y), (x,y), 1000, LOSS_FUNCTIONS['mse'], converter, 0.01)

    
    

