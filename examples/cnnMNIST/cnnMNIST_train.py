
#-----------------------------------------------------------------------------
# Copyright (c) 2019 Yang Zhenyu
# cdyangzhenyu@gmail.com
#
# Permission to use, copy, and modify this software and its documentation is
# hereby granted only under the following terms and conditions. Both the
# above copyright notice and this permission notice must appear in all copies
# of the software, derivative works or modified versions, and any portions
# thereof, and both notices must appear in supporting documentation.
# This software may be distributed (but not offered for sale or transferred
# for compensation) to third parties, provided such third parties agree to
# abide by the terms and conditions of this notice.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHORS, AS WELL AS THE UNIVERSITY
# OF BRITISH COLUMBIA DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
# INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO 
# EVENT SHALL THE AUTHORS OR THE UNIVERSITY OF BRITISH COLUMBIA BE LIABLE
# FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
# IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#---------------------------------------------------------------------------

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# Load the MNIST data set
mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)

# The basic MLP graph
x = tf.placeholder(tf.float32, shape=[None, 784], name="input")
x_image = tf.reshape(x, [-1,28,28,1])

fc_size = 20

w_c1 = tf.Variable(tf.truncated_normal([3, 3, 1, 3], stddev=0.1), name="w_c1")
b_c1 = tf.Variable(tf.constant(0.1, shape=[3]), name="b_c1")
w_c2 = tf.Variable(tf.truncated_normal([3, 3, 3, 8], stddev=0.1), name="w_c2")
b_c2 = tf.Variable(tf.constant(0.1, shape=[8]), name="b_c2")

w_fc1 = tf.Variable(tf.truncated_normal([7, 7, 8, fc_size], stddev=0.1), name="w_fc1")
b_fc1 = tf.Variable(tf.constant(0.1, shape=[fc_size]), name="b_fc1")
w_fc2 = tf.Variable(tf.truncated_normal([fc_size, 10], stddev=0.1), name="w_fc2")
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]), name="b_fc2")

h_p1 = tf.nn.max_pool(tf.nn.relu(tf.add(tf.nn.conv2d(x_image, w_c1, strides=[1, 1, 1, 1], padding='SAME'),b_c1)),ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

h_p2 = tf.nn.max_pool(tf.nn.relu(tf.add(tf.nn.conv2d(h_p1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2)),ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

keep_prob = tf.placeholder("float", name="keep_prob")
h_fc1 = tf.nn.dropout(tf.reshape(tf.nn.relu(tf.add(tf.nn.conv2d(h_p2, w_fc1, strides=[1, 1, 1, 1], padding='VALID'), b_fc1)),[-1, fc_size]), keep_prob)

y = tf.nn.softmax(tf.add(tf.matmul(h_fc1, w_fc2), b_fc2), name="output")

# The placeholder for the correct result
real_y = tf.placeholder(tf.float32, [None, 10], name="real_y")

# Loss function
cross_entropy = -tf.reduce_sum(real_y*tf.log(y))

# Optimization
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Correct Prediction
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(real_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialization
init = tf.global_variables_initializer()

saver = tf.train.Saver()

# Starting tf XLA session
with tf.Session() as session:

    # Training using MNIST dataset
    epochs = 20000
    session.run(init)
    for i in range(epochs):
        batch_x, batch_y = mnist_data.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_x, real_y: batch_y, keep_prob: 1.0})
            print "step %d, training accuracy %g"%(i, train_accuracy)
        session.run(train_step, feed_dict={x: batch_x, real_y: batch_y, keep_prob: 0.8})

    network_accuracy = session.run(accuracy, feed_dict={x: mnist_data.test.images, real_y: mnist_data.test.labels, keep_prob: 1.0})
    
    print('The accuracy over the MNIST data is {:.2f}%'.format(network_accuracy * 100))
    
    saver.save(session, "Model/model.ckpt")
