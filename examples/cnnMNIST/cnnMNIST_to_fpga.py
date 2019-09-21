import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import sys
sys.path.append('../../src')
import processMif as mif
import additionalOptions as options

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the 
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name="prefix", 
            op_dict=None, 
            producer_op_list=None
        )
    return graph

# We use our "load_graph" function
graph = load_graph("./Model/model.pb")

for op in graph.get_operations():
    print(op.name)  

# Get the model nodes
x = graph.get_tensor_by_name('prefix/input:0')
y = graph.get_tensor_by_name('prefix/output:0')
w_c1 = graph.get_tensor_by_name('prefix/w_c1:0')
b_c1 = graph.get_tensor_by_name('prefix/b_c1:0')
w_c2 = graph.get_tensor_by_name('prefix/w_c2:0')
b_c2 = graph.get_tensor_by_name('prefix/b_c2:0')

w_fc1 = graph.get_tensor_by_name('prefix/w_fc1:0')
w_fc2 = graph.get_tensor_by_name('prefix/w_fc2:0')
b_fc1 = graph.get_tensor_by_name('prefix/b_fc1:0')
b_fc2 = graph.get_tensor_by_name('prefix/b_fc2:0')

# Load the MNIST data set
mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)

test_image=123

with tf.Session(graph=graph) as session:
    with tf.device("device:XLA_CPU:0"):
        hp1 = tf.nn.max_pool(tf.nn.relu(tf.add(tf.nn.conv2d(tf.reshape(x, [-1,28,28,1]), w_c1, strides=[1, 1, 1, 1], padding='SAME'),b_c1)),ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        hp2 = tf.nn.max_pool(tf.nn.relu(tf.add(tf.nn.conv2d(hp1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2)),ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        h_fc1 = tf.nn.relu(tf.add(tf.nn.conv2d(hp2, w_fc1, strides=[1, 1, 1, 1], padding='VALID'), b_fc1))
        y = tf.nn.softmax(tf.add(tf.matmul(tf.reshape(h_fc1, [-1, 20]), w_fc2), b_fc2)) 
   
    ret = session.run(y, feed_dict={x: [mnist_data.test.images[test_image]]})

    print("Expected Result: "+str(np.argmax(mnist_data.test.labels[test_image])))
    print("Real Result: "+str(ret))

    # Creating memories for testing

    param1 = mnist_data.test.images[test_image]
    param0 = w_c1.eval()
    param2 = b_c1.eval()
    param3 = w_c2.eval()
    param4 = b_c2.eval()
    param5 = w_fc1.eval()
    param6 = b_fc1.eval()
    param7 = w_fc2.eval()
    param8 = b_fc2.eval()
    mif.createMem([param0,param1,param2,param3,param4,param5,param6,param7,param8])

#options.setUnrollThreshold(100000000)
