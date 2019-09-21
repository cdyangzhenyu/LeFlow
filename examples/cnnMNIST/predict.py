import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.platform import gfile
import numpy as np
import datetime

# Load the MNIST data set
mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.Session()
with gfile.FastGFile('./Model/model.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='prefix')

sess.run(tf.global_variables_initializer())

x = sess.graph.get_tensor_by_name('prefix/input:0')
keep_prob = sess.graph.get_tensor_by_name('prefix/keep_prob:0')
y = sess.graph.get_tensor_by_name('prefix/output:0')

#look the tensor name
#for op in sess.graph.get_operations():
#    print(op.name)

# real ret is 6
test_image=123

starttime = datetime.datetime.now()
#for i in range(10):
#  test_image=i
ret = sess.run(y, feed_dict={x: [mnist_data.test.images[test_image]], keep_prob: 1.0})
#  print("Expected Result: "+str(np.argmax(mnist_data.test.labels[test_image])))
#  print("Real Result: "+str(np.argmax(ret)))

endtime = datetime.datetime.now()

delta = (endtime - starttime).microseconds/1000.0

print("Expected Result: "+str(np.argmax(mnist_data.test.labels[test_image])))
print("Real Result: "+str(np.argmax(ret)))
print ret
print("Use time: %s ms" % str(delta))