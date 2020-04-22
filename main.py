import network
import numpy as np
from mlxtend.data import loadlocal_mnist
from time import time

def vector(l):
	v = np.zeros(10)
	v[l] = 1.0
	return v

def convert_labels(y):
	return np.array([vector(label) for label in y])

def main():
	dirp = "../MNIST_handwritten_digits/data/"

	train_img = dirp + "train_images"
	train_lab = dirp + "train_labels"

	x, y = loadlocal_mnist(images_path=train_img, labels_path=train_lab)
	y = convert_labels(y)

	arch = [x.shape[1], 300, 10]
	net = network.network(x, y, arch, normalize='yes')
	net.optimizer = net.AdamOptimizer	
	net.lossFunction = network.CrossEntropyLoss
	net.learning_rate = 0.01
	
	test_img = dirp + "test_images"
	test_lab = dirp + "test_labels"

	x_t, y_t = loadlocal_mnist(images_path=test_img, labels_path=test_lab)
	y_t = convert_labels(y_t)

	net.x_te = x_t
	net.y_te = y_t

	a = time()
	net.train(100)
	print(time() - a)
	#net.exportWeights("weights.txt")
	net.test(x_t, y_t)

if __name__ == "__main__":
	main()
