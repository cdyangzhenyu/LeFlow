import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.platform import gfile
import numpy as np
import datetime

tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('model_ckp_meta', './Model/model.ckpt.meta', 'model meta file path.')
tf.app.flags.DEFINE_string('model_dir', './Model', 'model dir.')

FLAGS = tf.app.flags.FLAGS

tf.reset_default_graph()
restore_graph = tf.Graph()

with tf.Session(graph=restore_graph) as restore_sess:
	restore_saver = tf.train.import_meta_graph(FLAGS.model_ckp_meta)
	restore_saver.restore(restore_sess,tf.train.latest_checkpoint(FLAGS.model_dir))

	#for op in restore_sess.graph.get_operations():
	#    print(op.name)

	x = restore_sess.graph.get_tensor_by_name('input:0')
	keep_prob = restore_sess.graph.get_tensor_by_name('keep_prob:0')
	y = restore_sess.graph.get_tensor_by_name('output:0')

	test_image=123

	mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)

	starttime = datetime.datetime.now()
	#for i in range(10):
	#  test_image=i
	ret = restore_sess.run(y, feed_dict={x: [mnist_data.test.images[test_image]], keep_prob: 1.0})
	#  print("Expected Result: "+str(np.argmax(mnist_data.test.labels[test_image])))
	#  print("Real Result: "+str(np.argmax(ret)))

	endtime = datetime.datetime.now()

	delta = (endtime - starttime).microseconds/1000.0

	print("Expected Result: "+str(np.argmax(mnist_data.test.labels[test_image])))
	print("Real Result: "+str(np.argmax(ret)))
	print ret
	print("Use time: %s ms" % str(delta))

