from typing import List, Tuple
from matrix import Matrix
from layer import Layer


class NeuralNetwork():
    def __init__(self):
        self.layers: List[Layer] = []

    
    def add_layer(self, neurons: int, activation,  inputs: int | None = None):
        if len(self.layers) == 0:
            self.layers.append(Layer(activation, inputs, neurons))
        
        else:
            inputs = self.layers[-1].neurons
            self.layers.append(Layer(activation, inputs, neurons))

    
    def predict(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs)
        return outputs       
    

    def train(self, train_set, test_set, epochs, loss_function, converter, learning_rate):
        x_train = train_set[0]
        y_train = train_set[1]

        x_test = test_set[0]
        y_test = test_set[1]

        loss = loss_function['function']
        loss_derivative = loss_function['derivative']

        for epoch in range(1, epochs+1):
            error = 0
            # good_predictions_train = 0
            #train
            for row in range(x_train.rows):

                outputs = self.predict(x_train.get_row(row, False))
                correct_outputs = y_train.get_row(row, False)

                error += sum(loss(correct_outputs, outputs).get_row(0))
                outputs_gradient = loss_derivative(correct_outputs, outputs)

                for layer in reversed(self.layers):
                    outputs_gradient = layer.backward(outputs_gradient, learning_rate)

                # if y_test.get_row(row, True) == converter.encode(converter.decode(outputs.get_row(0))):
                #     good_predictions += 1
        
            #test
            good_predictions_test = 0

            for row in range(x_test.rows):

                outputs = self.predict(x_test.get_row(row, False))
                correct_outputs = y_test.get_row(row, False)

                if y_test.get_row(row, True) == converter.encode(converter.decode(outputs.get_row(0))):
                    good_predictions_test += 1

            print(f'Epoch: {epoch}    Error: {round(error, 4)}    Accuracy: {round(good_predictions_test/x_test.rows, 3)}')
