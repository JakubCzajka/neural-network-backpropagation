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
![Image of simple feedforward network](md_resources/Colored_neural_network.svg.png)

As you can see, there are 3 types of layers in a feedforward neural network:

### Input layer

The first layer, colored red on the image above, is the input layer. There is always one input leayer in network, and dataset input serve as its inputs.

### Hidden layer

Network may contain zero or more hidden layers. Values of neurons in previous layer serve as inputs for hidden layer, and its outputs are connected to next layer in network as well.

### Output layer

The last layer in network is called output layer. Just like hidden layer, outputs of previous layer are connected as inputs to output layer, but values of neurons in output layer are not used in the network. Instead based on the neuron values user determines the output of whole network. In this program each neuron in output layer corresponds to one class from the dataset.

## How do neurons work

As mentioned earlier, each neuron has a set of connections and its value.

But how does each neuron figure its value? It's using a preety simple formulas:

$$ a = b + \sum_{i=1}^{n} w_{i}z_{i} $$
$$ z = f(a) $$

Breaking it into parts - $ a $ stands for neuron value before activation. $ b $ is the neuron bias - its property. The bias is added to te sum of products of previous layer's neuron values ($ z_{i} $) and their associated weights ($ w_{i} $). In the input layer instead of neuron values, $ z_{i} $ stands for input value taken from dataset. $ n $ is the number of neurons in previous layer (or number of inputs for current neuron).

Final value of neuron $ z $ is the value of neuron's activation function $ f $ at $ a $. Activation functions are usually nonlinear, like sigmoid functions or ReLU function.

### Vector approach

You might be familiar with the sum in first equation - it's preety similar to definition of dot product. - In fact, we can store layer's input and biases as vectors, and weights as matrix (in which each column represents weights for single neuron in layer). It allows us to calculate values for all neurons in layer using equations taht correspond to those above:

$$ A = Z_{i-1}W + B $$
$$ Z_{i} = f(A) $$

Where $ i $ stands for layer number, $ A $ is vector of neurons' values before activation, $ Z_{i-1} $ is vector outputs of previous layer (or dataset inputs), $ W $ is matrix of weights (as described above), $ B $ is vector of biases, and $ Z_{i} $ is vector of current layer neuron values.
$ f(A) $ is element-wise aplication of $ f $ for matrix $ A $

# Backpropagation

Great! Now we know how do neural network process data. But even more important is how they **learn** to correctly classify entries in datasets. One of the most basic algorithms used for neural network lerning is the **backpropagation** algorithm.

## Error function

First task is to evaluate the network on how well it does its job. In order to do so we pass network outputs and correct outputs of train dataset to **error function**. The higher the value of that error function the worse the network performed. One of the most commonly used error functions is the mean squared error function: $$ E_{n} = MSE(z_{n}, r_{n}) = \frac{(z_{n} - r_{n})^{2}}{2} $$ $ E_{n} $ is error of given neuron in output layer, $ z_{n} $ is that neuron value and $ r_{n} $ is the expected value of that neuron. Error of the network $ E $ is sum of all errors of neurons in output layer.

## Gradient descent

In order to improve performance ot the network we want its error function to be as low as possible. We can do it by finding minimum of error function using for example method known as gradient descent.

Since value of error function depends on all parameters (biases and weights) in network, we have to calculate error function derivatives for all of them.

### Output layer

Let's start by explaining how it works for neurons in output layer, since we've already calculated their errors.

In order to calculate $ \frac{dE}{dw_{i}} $ (derivative of error function with respect to any input weight) of we will apply the chain rule: $$ \frac{dE}{dw_{i}} = \frac{dE}{dz_{n}} * \frac{dz_{n}}{da_{n}} * \frac{da_{n}}{dw_{i}} $$ Now we have to find the factors of above equation.

The first factor $ \frac{dE}{dz_{n}} $ tells how much does the error changw with respect to neuron output. Since $ E = E_{1} + E_{2} + ...  + E_{n} + ... + E_{N} $ and $ z_{n} $ is present only in $ E_{n} $, then $ \frac{dE}{dz_{n}} = \frac{dE_{n}}{dz_{n}} $. This is called output gradient of a neuron. When using mean square error function it will be simply $ MSE'(z_{n}, r_{n}) = z_{n} - r_{n} $.

The second factor, $ \frac{dz_{n}}{da_{n}} $ is preety simple to calculate, because $ z_{n} = f(a_{n}) $, then $ \frac{dz_{n}}{da_{n}} = f'(a_{n}) $

$ \frac{da_{n}}{dw_{i}} $ Is trivial to calculate as well - because $ a_{n} = w_{1}z_{1} + w_{2}z_{2} + ... + w_{i}z_{i} + ... + w_{I}z_{I} + b $, then $ \frac{da_{n}}{dw_{i}} = z_{i} $ 

That concludes calculating $ \frac{dE}{dw_{i}} $, in order to calculate $ \frac{dE}{db} $ we will once again use the chain rule: $$ \frac{dE}{db} = \frac{dE}{dz_{n}} * \frac{dz_{n}}{da_{n}} * \frac{da_{n}}{db} $$.

As you can see, only the last factor is different: $ \frac{da_{n}}{db} $. We can calculate it, since $ a_{n} = w_{1}z_{1} + w_{2}z_{2} + ... + w_{i}z_{i} + ... + w_{I}z_{I} + b $, then $ \frac{da_{n}}{db} = 1 $

### Backpropagating the error

Integrals calculated abowe will apply to neurons in different layers as well. But there is one problem: we don't have $ \frac{dE}{dz_{n}} $ calculated, since hidden and input neurons' values are not **explicitly** in MSE formula. We know that output of those neurons do contribute to values of output neurons. In general: $$ \frac{dE}{dz_{n}} = \sum_{k=1}^{K}\frac{dE_{k}}{dz_{n}} $$ where $ E_{k} $ is the error of neuron in next layer, and $ K $ is number of neurons in next layer.

For any neuron in next layer we can calculate $ \frac{dE_{k}}{dz_{n}} $ by applying the chain rule: $$ \frac{dE_{k}}{dz_{n}} = \frac{dE_{k}}{dz_{k}} * \frac{dz_{k}}{da_{k}} * \frac{da_{k}}{dz_{n}} $$ 

$ \frac{dE_{k}}{dz_{k}} * \frac{dz_{k}}{da_{k}} $ is known for the output neurons, so we have to find the value of $ \frac{da_{k}}{dz_{n}} $.

Since $ a_{k} = w_{1}z_{1} + w_{2}z_{2} + ... + w_{n}z_{n} + ... + w_{I}z_{I} + b $, then $ \frac{da_{k}}{dz_{n}} = w_{n} $  that is weight connecting selected neuron and the neuron in next layer.

Now we know that $$ \frac{dE}{dz_{n}} = \sum_{k=1}^{K}\frac{dE_{k}}{dz_{n}} = \sum_{k=1}^{K}\frac{dE_{k}}{dz_{k}} * \frac{dz_{k}}{da_{k}} * w_{n} $$

$\frac{dE_{k}}{dz_{k}}$ (and $\frac{dz_{k}}{da_{k}}$) are known for neurons in output layer, so we have to propagate the error starting from the last hidden layer. That way those factors will be known for more and more layers sequentially.

## Vector approach

Since we don't want to represent single neurons in code, but rather whole layers, we have to adjust above equations, so that they operate on vectors/matrices that are stored in the layer class.

Each layer (especially the output layer) reveives vector of ouptut gradeints $ \frac{dE}{dZ} $ (vector of $ \frac{dE_{n}}{dz_{n}} $) as parameter in backpropagation method.

In order to calculate $ \frac{dz_{n}}{da_{n}} $ we have to apply derivative of layer's activation function element-wise on our vector of pre-activation values: $ \frac{dZ}{dA} = f'(A) $

$ \frac{dE}{dA} = \frac{dE}{dZ} * \frac{dZ}{dA} $ is the bias nudges and base for weight nudges. **Please note that this is element-wise multiplication!!**

In order to calculate input gradient, that will be passed to previous layer, we have to multiply $ \frac{dE}{dZ} * \frac{dZ}{dA} * W^{t} $ - this will result in a vector containing dot products of weights and input errors of each neuron.

We do also have vector of inputs $ X $, that will be useful in calculating weight nudges, if we multiply the $ X^{t} $ by $ \frac{dE}{dA} $.

Before subtracting nudges from biases and weights, we have to multiply them by a gradient descent parameter known as **learning rate**. It is a small float.