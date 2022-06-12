from matrix import Matrix, split_dataset
from converter import Converter

from activation_functions import FUNCTIONS as ACTIVATION_FUNCTIONS
from loss_functions import FUNCTIONS as LOSS_FUNCTIONS
from shell import NetworkShell



if __name__ == '__main__':
    NetworkShell().cmdloop()

    # dataset, converter = iris_data()

    # train_data, test_data = split_dataset(dataset, 0.7)

    # network = NeuralNetwork.create_from_json('resources/network_structure.json', ACTIVATION_FUNCTIONS, dataset[0].columns, dataset[1].columns)
    # network.train(train_data, test_data, 10000, LOSS_FUNCTIONS['mse'], converter, 0.1)

    
    

