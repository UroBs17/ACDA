import numpy as np
from errors import Errors as errors
from activations import Activations as activations

class Network:
    def __init__(self, inputs, targets, hidden_layer_neurons = 2, learning_rate = 0.5):
        self.inputs = inputs
        self.targets = targets
        self.hidden_layer_neurons=hidden_layer_neurons
        self.learning_rate=learning_rate
        self.weights_layer_1=np.array([[np.random.random_sample() for i in range(len(inputs[0]))]for j in range(hidden_layer_neurons+1)])
        self.weights_layer_2=np.array([[np.random.random_sample()] for i in range(hidden_layer_neurons+1)])
    def fit(self, epochs):
        for i in range(epochs):
            first_layer = np.dot(self.inputs, self.weights_layer_1)
            for i in range(first_layer.shape[0]):
                for j in  range(first_layer.shape[1]):
                    first_layer[i][j]=activations.sigmoid(first_layer[i][j])
            second_layer = np.dot(first_layer,self.weights_layer_2)
            for i in range(second_layer.shape[0]):
                for j in  range(second_layer.shape[1]):
                    second_layer[i][j]=activations.sigmoid(second_layer[i][j])
            error= np.subtract(second_layer,self.targets)
            print(error)
            
        
if __name__ == '__main__':
    input=[
        [0,1,1],
        [0,0,1],
        [0,1,1]
    ]
    target=[
        [0],
        [1],
        [0]
    ]
    nn = Network(input,target)
    nn.fit(1)