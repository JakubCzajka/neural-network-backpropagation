from matrix import Matrix, split_dataset
from converter import Converter

from activation_functions import FUNCTIONS as ACTIVATION_FUNCTIONS
from loss_functions import FUNCTIONS as LOSS_FUNCTIONS
import pandas as pd

from network import NeuralNetwork


def iris_data():
    dataset = pd.read_csv('resources/iris.data', delimiter=',')
    return Matrix.read_dataset(dataset), Converter(dataset)


if __name__ == '__main__':
    
    dataset, converter = iris_data()

    train_data, test_data = split_dataset(dataset, 0.7)

    network = NeuralNetwork()
    network.add_layer(3, ACTIVATION_FUNCTIONS['sigmoid'], dataset[0].columns)
    network.add_layer(7, ACTIVATION_FUNCTIONS['sigmoid'])
    network.add_layer(dataset[1].columns, ACTIVATION_FUNCTIONS['sigmoid'])

    network.train(train_data, test_data, 10000, LOSS_FUNCTIONS['mse'], converter, 0.1)

    
    

