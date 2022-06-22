# Machine learning
Machine learning is a part of computer science dedicated to creating and understanding structures and algorithms that can improve themselves - in a process similar to humans learining things as they grow. This field of study is closely related to statistics, and finds applications in many not computers-related fields such as medicine, astronomy or agriculture.

There are 3 main approaches to machine learning:
- Supervised learning
- Unsupervised learning
- Reinforcement learning

## Supervised learning
In this approach the model is built upon dataset that contains inputs and desired outputs. The model is supposed to discover rules that connect inputs and outputs. This requirement renders a huge drawback - training data has to be labeled by humans.

This is the approach that will be used in this program. More specifically the model will be able to learn using datasets that contain numerical-only inputs and one output - record's class. Example of such dataset is the iris dataset.

# Neural networks
One type of models used in ML are **neural networks**. They are structrures loosely based on animal brains. Single neural network consists of many even smaller structures called neurons. They are based on biological brains' **neurons**. Neurons in network are connected together and they transmit signals between each other.

In their simplest form neurons form themselves in **layers**, such that neurons in single layer can tereat only neurons in previous layer as their inputs. If the connections between neurons don't form a cycle the network is called **feedforward network**.

## Feedforward networks
Below you can see a image of simple feedforward network:
![](md_resources/Colored_neural_network.svg.png)

As you can see, there are 3 types of layers in a feedforward neural network:
### Input layer
The first layer, colored red on the image above, is the input layer. There is always one input leayer in network, and dataset input serve as its inputs.
### Hidden layer
Network may contain zero or more hidden layers. Values of neurons in previous layer serve as inputs for hidden layer, and its outputs are connected to next layer in network as well.
### Output layer
The last layer in network is called output layer. Just like hidden layer, outputs of previous layer are connected as inputs to output layer, but values of neurons in output layer are not used in the network. Instead based on the neuron values user determines the output of whole network. In this program each neuron in output layer corresponds to one class from the dataset.

## How do neurons work
# Backpropagation

