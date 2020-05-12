import numpy as np
from measures import Measures

class activations:
	def sigmoid(x):
		return 1 / (1 + np.exp(-x))
	def sigmoid_derivative(x):
		return x*(1 - x)

class errors:
	def mse(real, predicted):
		if len(real) == len(predicted):
			total = 0
			for i in range(len(real)):
				total += (((real[i] - predicted[i])**2)/len(real))
			return total**(0.5)

class network:
    def __init__(self, inputs, targets, hidden_layer_neurons = 2, learning_rate = 0.5):
        self.inputs = inputs
        self.targets = targets
        self.hidden_layer_neurons = hidden_layer_neurons
        self.learning_rate = learning_rate
        self.weights_layer_1 = np.random.rand(hidden_layer_neurons + 1, inputs.shape[1])
        self.weights_layer_2 = np.random.rand(hidden_layer_neurons + 1, 1)
        
    def fit(self, epochs):
        for epoch in range(epochs):
            #Forward pass
            inputs_dot_weights1 = np.dot(self.inputs,self.weights_layer_1)
            activation_inputs_dot_weights1 = activations.sigmoid(inputs_dot_weights1)
            hidden_dot_weights2 = np.dot(activation_inputs_dot_weights1,self.weights_layer_2)
            activation_hidden_dot_weights2 = activations.sigmoid(hidden_dot_weights2)
            #Backward Pass
            ##Output layer
            error = self.targets - activation_hidden_dot_weights2
            delta_activation_output = activations.sigmoid_derivative(activation_hidden_dot_weights2)
            error_correction_output = delta_activation_output * error
            ##Hidden layer
            error_output_dot_weights2T = np.dot(error_correction_output, self.weights_layer_2.T)
            activation_error_hidden_layer = activations.sigmoid_derivative(activation_inputs_dot_weights1)
            error_correction_hidden_layer = activation_error_hidden_layer * error_output_dot_weights2T
            #Adjusts weights
            errors_through_layer2 = np.dot(activation_inputs_dot_weights1.T,error_correction_output)
            self.weights_layer_2 += errors_through_layer2 * self.learning_rate
            errors_through_layer1 = np.dot(self.inputs.T, error_correction_hidden_layer)
            self.weights_layer_1 += errors_through_layer1 * self.learning_rate
    
    def predict(self, inputs):
        inputs_dot_layer1 = np.dot(inputs, self.weights_layer_1)
        activate_i_dot_l1 = activations.sigmoid(inputs_dot_layer1)
        layer1_dot_layer2 = np.dot(activate_i_dot_l1, self.weights_layer_2)
        return activations.sigmoid(layer1_dot_layer2)
if __name__ == '__main__':
    
    for i in range(6):
        f= open("data"+str(i+1)+".txt","r")
        inputs = []
        targets = []
        n = int(f.readline())
        for line in range(n):
            l = list(map(int, f.readline().strip().split()))
            inputs.append(l[:-1])
            targets.append([l[-1]])
        inputs = np.array(inputs)
        targets = np.array(targets)
        nn = network(inputs,targets)
        nn.fit(10000)
        pred = nn.predict(inputs)
        m = Measures(targets,pred)
        print("---------------")
        print("precision", m.precision())
        print("recall", m.recall())
        print("f1", m.f1())
        print("accuracy", m.accuracy())
        print("confusion matrix")
        for i in m.cmatrix():
            print(i)
        