import numpy as np

class Measures():
	def __init__(self, real, pred):
		self.real = real
		self.pred = pred

		self.tn = 0
		self.fp = 0
		self.fn = 0
		self.tp = 0

		for i in range(len(self.real)):
			if self.real[i] <= 0 and self.pred[i] <= 0.1:
				self.tn = self.tn + 1
			if self.real[i] <= 0 and self.pred[i] > 0.9:
				self.fp = self.fp + 1
			elif self.real[i] > 0 and self.pred[i] <= 0.1:
				self.fn = self.fn + 1
			elif self.real[i] > 0 and self.pred[i] > 0.9:
				self.tp = self.tp + 1

	def precision(self):
		return self.tp / (self.tp + self.fp)

	def recall(self):
		return self.tp / (self.tp + self.fn)

	def f1(self):
		p = self.precision()
		r = self.recall()

		return 2 * (p * r) / (p + r)

	def accuracy(self):
		return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

	def cmatrix(self):
		cm = [
			[self.tp, self.fp],
			[self.fn, self.tn]
		]

		return cm