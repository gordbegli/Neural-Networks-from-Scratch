import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs): 
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs): #ReLU's in sequence can approximate any curve
        self.output = np.maximum(0, inputs)

class Activation_Softmax:#softmax (exp --> norm) amplifies differences between output neuron values
    def forward(self, inputs): #inputs is a batch!
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) #to prevent overflow here we subtract largest val from all vals, this doesn't change the resulting distribution surprisingly
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) 
        #some people use one hot, some people use int values for labeling
        if len(y_true.shape) == 1: 
            correct_confidences = y_pred_clipped[range(samples), y_true]


X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])