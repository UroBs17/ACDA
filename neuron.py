import matplotlib.pyplot as plt    
import numpy as np
from miscellaneous import Miscellaneous
from measures import Measures
class Neuron():
	def __init__(self, weights, bias=False, theta=0):
		self.weights = weights
		self.bias = bias 
		self.theta = theta
	def process(self, data):  
		if self.bias:
			data =np.insert(data, 0, 1)
		self.v = sum([data[i] * self.weights[i] for i in range(len(data))])
		return self.v
	def activate(self, condition):
	z= condition(self.v, self.theta)
	self.z=z
	return z
if __name__ == "__main__":
	dataset= np.array([[0,0],[0,1],[1,0],[1,1]])
	weights = np.array([1, 1])
	theta = 1
	neuron = Neuron(weights, bias = False,theta = theta)
	real = np.array([0,1,1,1])
	pred = np.array([])
	for data in dataset:
		neuron.process(data)
		neuron.activate(lambda v, t: 1 if v >= t else 0)
		pred= np.append(pred,neuron.z)
	print(pred)
	m=Measures(real,pred)
	Miscellaneous.printcmatrix(m.cmatrix())
	print(m.precision())
	print(m.recall())
	print(m.f1())
	print(m.accuracy())
	x1 = np.linspace(-3, 3, 10) 
	x2 = np.linspace(-3, 3, 0)
	v = -(weights[0] / weights[1]) * x1 + theta / weights[1]
	plt.fill_between(x1, v, 3, where = v < 3, color = 'g', alpha = 0.5)
	ones = dataset[np.where(real > 0)]
	zeros = dataset[np.where(real <= 0)]
	plt.plot(ones[:,0],ones[:,1], 'gs')
	plt.plot(zeros[:,0],zeros[:,1], 'ro')
	plt.axis([-1 ,2 ,-1, 2])
	plt.show()