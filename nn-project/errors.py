class Errors:
	def mse(real, predicted):
		if len(real) == len(predicted):
			total = 0
			for i in range(len(real)):
				total += ((real[i] - predicted[i])**2)
			total = total/n
			return total**(0.5)