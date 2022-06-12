import cmd

from matrix import Matrix, split_dataset
from converter import Converter
from network import NeuralNetwork
from activation_functions import FUNCTIONS as ACTIVATION_FUNCTIONS
from loss_functions import FUNCTIONS as LOSS_FUNCTIONS
import pandas as pd


class NetworkShell(cmd.Cmd):
    intro = 'Neural network shell.\nType help or ? to list commands'
    prompt = '(state: hello)'
    results = None
    network = None
    dataset = None
    test_dataset = None
    converter = None

    def do_load_dataset(self, line):
        'Load dataset from CSV file: load_dataset <filepath> <train_rows_percentage>'

        if self.network:
            print('Impossible to load dataset with network created')
            return

        file, ratio = [s for s in line.split(':')]
        ratio = float(ratio)

        dataset = pd.read_csv(file, delimiter=',')
        self.dataset = Matrix.read_dataset(dataset)
        self.converter = Converter(dataset)

        self.dataset, self.test_dataset = split_dataset(self.dataset, ratio)

        self.prompt = '(state: dataset loaded)'


    def do_create_network(self, file):
        'Create network based on json configuration file: create_network <filepath>'

        if self.dataset is None:
            print('Load dataset before creating a network')

        else:
            self.network = NeuralNetwork.create_from_json(file,
                                                            ACTIVATION_FUNCTIONS,
                                                            self.dataset[0].columns,
                                                            self.dataset[1].columns)
            self.prompt = '(state: network created)'
        
    
    def do_train(self, line):
        'Train network on loaded dataset: train <loss function>:<maximum number of epochs>:<learning rate>:<desired accuracy>:<verbose>'
        if self.dataset is None:
            print('No dataset loaded')
            return
        
        if self.network is None:
            print('No network created')
            return

        loss_function, epochs, learning_rate, desired_accuracy, verbose = [s for s in line.split(':')]
        epochs = int(epochs)
        learning_rate = float(learning_rate)
        desired_accuracy = float(desired_accuracy)
        verbose = verbose in ['Yes', 'yes', 'y', 'Y']

        self.results = self.network.train(self.dataset,
                                            self.test_dataset,
                                            epochs,
                                            LOSS_FUNCTIONS[loss_function],
                                            self.converter,
                                            learning_rate,
                                            desired_accuracy,
                                            verbose)
        self.prompt = '(state: trained)'


    def postcmd(self, stop: bool, line: str) -> bool:
        print()
        return super().postcmd(stop, line)