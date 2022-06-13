from shell import NetworkShell
import cmd
import pickle
import os

from matrix import Matrix, split_dataset
from converter import Converter
from network import NeuralNetwork
from activation_functions import FUNCTIONS as ACTIVATION_FUNCTIONS
from loss_functions import FUNCTIONS as LOSS_FUNCTIONS
import pandas as pd
import matplotlib.pyplot as plt



if __name__ == '__main__':
    NetworkShell().cmdloop()
    # dataset = pd.read_csv('resources/iris.data', delimiter=',')
    # converter = Converter(dataset)
    # dataset = Matrix.read_dataset(dataset)
    

    # dataset, test_dataset = split_dataset(dataset, 0.7)

    # network = NeuralNetwork.create_from_json('resources/network_structure.json',
    #                                                 ACTIVATION_FUNCTIONS,
    #                                                 dataset[0].columns,
    #                                                 dataset[1].columns)
       

    # epochs = int(200)
    # learning_rate = float(0.05)
    # desired_accuracy = float(0.9)
    # verbose = True

    # results = network.train(dataset,
    #                         test_dataset,
    #                         epochs,
    #                         LOSS_FUNCTIONS['mse'],
    #                         converter,
    #                         learning_rate,
    #                         desired_accuracy,
    #                         verbose)

  
    # df = pd.DataFrame(results)
    # plt.figure(1)
    # plt.plot(df['epoch'], df['accuracy'], 'k-')
    # plt.show()
    # pass
