import numpy as np
from sklearn.utils import shuffle

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
	return sigmoid(x) * (1.0 - sigmoid(x))	

class QuadraticLoss():

	def Loss(a, y): 		
		return 0.5 * np.linalg.norm(a-y)**2
	
	def Grad(a, y, z):
		return (a - y) * sigmoid_prime(z)

class CrossEntropyLoss():

	def Loss(a, y):
		return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a))) 		

	def Grad(a, y, z):
		return (a - y)

class network():
	
	def __init__(self, x, y, arch, normalize='no', init=''):
		self.x = x.astype(float)
		self.y = y.astype(float)
		self.x_te = None
		self.y_te = None
		self.arch = arch
		self.optimizer = self.SGD
		self.lossFunction = QuadraticLoss
		self.learning_rate = 0.001
		self.first_moment_b = np.array([np.zeros(self.arch[i]) for i in range(1, len(self.arch))])
		self.first_moment_w = np.array([np.zeros((self.arch[i], self.arch[i+1])) for i in range(len(self.arch) - 1)])
		self.second_moment_b = np.array([np.zeros(self.arch[i]) for i in range(1, len(self.arch))])
		self.second_moment_w = np.array([np.zeros((self.arch[i], self.arch[i+1])) for i in range(len(self.arch) - 1)])	
		self.beta1 = 0.9
		self.beta2 = 0.999

		if normalize == 'yes':
			#mean_image = np.sum(self.x, axis=0) / self.x.shape[0]
			#self.x -= mean_image		
			self.x /= 255
	
		self.weights = np.array([np.random.randn(self.arch[i], self.arch[i+1]) / np.sqrt(self.arch[i+1])
								 for i in range(len(self.arch)-1)])
		self.biases = np.array([np.random.randn(self.arch[i]) for i in range(1, len(self.arch))]) 

	def SGD(self, l, g, wm, i):
		self.weights[wm - 1] -= self.learning_rate * np.dot(l[i-2].T, g)
		self.biases[wm - 1] -= self.learning_rate * np.sum(g, axis=0)

	def AdamOptimizer(self, l, g, wm, i):
		dw = np.dot(l[i-2].T, g)
		db = np.sum(g, axis=0)
			
		self.first_moment_w[wm - 1] = self.beta1 * self.first_moment_w[wm - 1] + (1 - self.beta1) * dw
		self.first_moment_b[wm - 1] = self.beta1 * self.first_moment_b[wm - 1] + (1 - self.beta1) * db
		self.second_moment_w[wm - 1] = self.beta2 * self.second_moment_w[wm - 1] + (1 - self.beta2) * dw * dw
		self.second_moment_b[wm - 1] = self.beta2 * self.second_moment_b[wm - 1] + (1 - self.beta2) * db * db
		
		first_unbiased_w = self.first_moment_w[wm - 1] / (1 - self.beta1 ** self.epoch)
		first_unbiased_b = self.first_moment_b[wm - 1] / (1 - self.beta1 ** self.epoch)
		second_unbiased_w = self.second_moment_w[wm - 1] / (1 - self.beta2 ** self.epoch)
		second_unbiased_b = self.second_moment_b[wm - 1] / (1 - self.beta2 ** self.epoch)

		self.weights[wm - 1] -= self.learning_rate * first_unbiased_w / (np.sqrt(second_unbiased_w) + 1e-7)
		self.biases[wm - 1] -= self.learning_rate * first_unbiased_b / (np.sqrt(second_unbiased_b) + 1e-7)

	def backprop(self, l, y):
		i = len(l) - 1
		grad = self.lossFunction.Grad(l[-1], y, l[-2])
		for wm in range(len(self.arch) - 1, 0, -1):
			if wm != len(self.arch) - 1:
				grad *= sigmoid_prime(l[i-1])

			next_grad = np.dot(grad, self.weights[wm - 1].T)
			
			self.optimizer(l, grad, wm, i)

			grad = next_grad
			i -= 2

	def train(self, epochs):
		self.optimizerBackup = self.optimizer
		for i in range(epochs):
			training_data_x, training_data_y = shuffle(self.x, self.y)
			mini_batches_x = [training_data_x[k:k+5000] for k in range(0, self.x.shape[0], int(self.x.shape[0] / 10))]
			mini_batches_y = [training_data_y[k:k+5000] for k in range(0, self.x.shape[0], int(self.x.shape[0] / 10))]
			#mini_batches_x = [self.x]
			#mini_batches_y = [self.y]
			self.epoch = i + 1
			for d in range(len(mini_batches_x)):
				x = mini_batches_x[d]
				l = [x]
				for layer in range(len(self.arch) - 1):
					x = np.dot(x, self.weights[layer]) + self.biases[layer]
					l.append(x)
					x = sigmoid(x)
					l.append(x)
			
				print("Epoch %d - Loss %f " %(i, self.lossFunction.Loss(x, mini_batches_y[d])))
				self.backprop(l, mini_batches_y[d])
			if i % 5 == 0:
				self.test(self.x_te, self.y_te)

	def exportWeights(self):
		pass

	def test(self, x_te, y_te):
		x = x_te
		for layer in range(len(self.arch) - 1):
			x = sigmoid(np.dot(x, self.weights[layer]) + self.biases[layer])
		pred = np.argmax(x, axis=1)
		true = np.argmax(y_te, axis=1)

		correct = np.sum(pred == true)

		print(float(correct) / float(true.shape[0]))

"""
	#print(self.lossFunction.Grad(x, self.y, self.y))	
					#print("x: {0}\ny: {1}\ngrad: {2}\nw: {3}\nb: {4}\nl[i]: {5}".format(self.x.shape,self.y.shape,
			#		self.lossFunction.Grad(x, self.y, self.y).shape,self.weights[1].shape, self.biases[1].shape,
			#		l[-3].shape))

def relu(x):
	return np.maximum(0, x) 

def relu_prime(x):
	return (x > 0) * 1

		#wx + b = c -> relu(c) = d -> wd + b = z -> sigmoid(z) = a (10 x 1) 

		# x , c , d , z , a
		# 0 , 1 , 2 , 3 , 4
"""
