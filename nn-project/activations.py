from math import e
class Activations:
	def sigmoid(x):
		return 1 / (1 + e**(-x))
	def sigmoid_derivative(x):
		return sigmoid(x)*(1 - sigmoid(x))
	