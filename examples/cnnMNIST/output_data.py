from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

test_image=123
mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)

data = np.array([mnist_data.test.images[test_image]])

print data.tolist()