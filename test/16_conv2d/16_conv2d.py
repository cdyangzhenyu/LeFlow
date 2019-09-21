#-----------------------------------------------------------------------------
# Copyright (c) 2018 Daniel Holanda Noronha, Bahar Salehpour, Steve Wilton
# danielhn<at>ece.ubc.ca
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
import string

size = 5
outputs = 2

X = tf.placeholder(tf.float32, [1, size,size,1])
weights = tf.placeholder(tf.float32, [3,3,1,outputs])
w1 = tf.placeholder(tf.float32, [3,3,2,outputs])

x=[1,0,1,1,-1,1,1,0,-1,0,1,0,0,-1,0,1,0,1,1,-1,-1,0,-1,1,1]
w11=[1,-1,1,0,0,0,1,0,0,1,0,-1,0,1,1,0,0,-1]
w2=[1,-1,1,0,0,0,1,-1,1,0,0,0,1,-1,1,0,0,0,0,1,1,0,0,-1,0,1,1,0,0,-1,1,0,0,1,-1,0]
in_x=np.reshape(np.array(x).transpose(),[1,size,size,1])

in_weights = np.reshape(np.array(w11).transpose(),[3,3,1,outputs])
in_w1 = np.reshape(np.array(w2).transpose(),[3,3,2,outputs])

print in_x.reshape(1,5,5,1)
print "======="
print in_weights.reshape(3,3,1,2)
print "======="
print in_w1.reshape(3,3,2,2)
print "======="

y1 = tf.nn.conv2d(X, weights, strides=[1, 1, 1, 1], padding='SAME')
y = tf.nn.conv2d(y1, w1, strides=[1, 1, 1, 1], padding='SAME')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    r1 = sess.run(y1, feed_dict={X: in_x, weights: in_weights})
    result = sess.run(y, feed_dict={X: in_x, weights: in_weights, w1: in_w1})
    print r1.reshape(1,size,size,2)
    print "======="
    print result.reshape(1,size,size,2)
