import numpy as np

class Net():
	def predict(self, x):
		return self.activate(self, np.dot(x, self.W))

	def saveweights(self, fname):
		np.save("%s.npy" % fname, self.W)

	def loadweights(self, fname):
		self.W = np.load("%s.npy" % fname)

	def activate(self, v):
		if v >= 0:
			return 1
		elif v < 0:
			return -1

	activate = np.vectorize(activate)

class Hebb(Net):
	def fit(self, x, t, max_epochs = 50):
		self.W = np.zeros(x.shape[0] - 1)

		for i in range(max_epochs):
			for sample, target in zip(x, t):
				v = np.dot(sample, self.W)
				
				self.W = self.W + sample * target

class Perceptron(Net):
	def predict(self, x):
		return self.activate(self, np.dot(x, self.W), self.theta)

	def fit(self, x, t, alpha = 1, theta = 1):
		self.W = np.random.random(x.shape[0] - 1)
		self.W_old = np.zeros(x.shape[0] - 1)
		self.theta = theta

		conv_epoch = 0

		while True:
			for sample, target in zip(x, t):
				v = np.dot(sample, self.W)

				z = self.activate(self, v, theta)
				
				if z != target:
					self.W = self.W_old + alpha * target * sample

			conv_epoch = conv_epoch + 1

			if np.max(np.abs(self.W - self.W_old)) == 0:
				break
			else:
				self.W_old = self.W

		return conv_epoch

	def activate(self, v, theta):
		if v > theta:
			return 1
		elif -theta <= v and v <= theta:
			return 0
		elif v < -theta:
			return -1

	activate = np.vectorize(activate)

class Adaline(Net):
	def fit(self, x, t, alpha = 1, tolerance = 1e-6):
		self.W = np.random.random(x.shape[0] - 1)
		self.W_old = np.zeros(x.shape[0] - 1)

		conv_epoch = 0

		while True:
			for sample, target in zip(x, t):
				v = np.dot(sample, self.W)

				z = self.activate(self, v)
				
				if z != target:
					self.W = self.W_old + alpha * (target - z) * sample

			conv_epoch = conv_epoch + 1

			if np.max(np.abs(self.W - self.W_old)) <= tolerance:
				break
			else:
				self.W_old = self.W

		return conv_epoch

def getR(x):
	return np.sum(np.dot(x.T, x_train), axis = 0) / len(x)

if __name__ == '__main__':

	#bipolar inputs
	x_train = np.array([     #x1   x2     b   
							[ -1,  -1,    1], #1
							[ -1,   1,    1], #2
							[  1,  -1,    1], #3
							[  1,   1,    1]  #4
					  ])

	#bipolar targets
	y_train = np.array([	#t
							-1, #1
							-1, #2
							-1, #3
							 1  #4
						])

	HebbNet = Hebb()
	HebbNet.fit(x_train, y_train, 10)

	y_pred = HebbNet.predict(x_train)
	print("Real:", y_train, "Predicted:", y_pred)

	HebbNet.saveweights("weights_hebb")


	HebbNet1 = Hebb()
	HebbNet1.loadweights("weights_hebb")

	y_pred = HebbNet1.predict(x_train)
	print("Real:", y_train, "Predicted:", y_pred)


	print("\n")


	PerceptronNet = Perceptron()

	alphas = [0.1 * i for i in range(1, 11)]
	theta = 1

	for alpha in alphas:
		converged = PerceptronNet.fit(x_train, y_train, alpha, theta)
		y_pred = PerceptronNet.predict(x_train)

		print("a = %0.1f" % alpha, "e = %2d" % converged, "y_train:", y_train, "y_pred:", y_pred, "equals?", np.array_equal(y_train, y_pred))

	print("\n")

	np.random.seed(1)
	AdalineNet = Adaline()

	alphas = [0.1 * i for i in range(1, 11)]

	tolerances = [3, 1.5, 1e-2, 1e-12]

	for alpha in alphas:
		for tolerance in tolerances:
			converged = AdalineNet.fit(x_train, y_train, alpha, tolerance)
			y_pred = AdalineNet.predict(x_train)

			print("a = %0.1f" % alpha, "e = %2d" % converged, "y_train:", y_train, "y_pred:", y_pred, "equals?", np.array_equal(y_train, y_pred))

	print("\n")

	AdalineNet1 = Adaline()
	R = getR(x_train[:,:1])

	alpha = np.max(R) / 2

	converged = AdalineNet1.fit(x_train, y_train, alpha, 1e-6)
	y_pred = AdalineNet1.predict(x_train)
	print("a = %0.1f" % alpha, "e = %2d" % converged, "y_train:", y_train, "y_pred:", y_pred, "equals?", np.array_equal(y_train, y_pred))