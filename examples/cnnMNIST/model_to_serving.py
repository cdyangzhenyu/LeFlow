import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.platform import gfile
import numpy as np
import datetime
import os

tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('model_ckp_meta', './Model/model.ckpt.meta', 'model meta file path.')
tf.app.flags.DEFINE_string('model_dir', './Model', 'model dir.')
tf.app.flags.DEFINE_string('export_path_base', './Model/test', 'export model path.')

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

	tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
	tensor_info_y = tf.saved_model.utils.build_tensor_info(y)
	tensor_info_keep_prob = tf.saved_model.utils.build_tensor_info(keep_prob)

	prediction_signature = (
	  tf.saved_model.signature_def_utils.build_signature_def(
	      inputs={'x': tensor_info_x, 'keep_prob': tensor_info_keep_prob},
	      outputs={'y': tensor_info_y},
	      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

	legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

	export_path = os.path.join(
	  tf.compat.as_bytes(FLAGS.export_path_base),
	  tf.compat.as_bytes(str(FLAGS.model_version)))
	print ('Exporting trained model to', export_path)

	builder = tf.saved_model.builder.SavedModelBuilder(export_path)

	builder.add_meta_graph_and_variables(
	  restore_sess, [tf.saved_model.tag_constants.SERVING],
	  signature_def_map={
	      'predict_images':
	          prediction_signature,
	      # tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
	      #     classification_signature,
	  },
	  legacy_init_op=legacy_init_op)

	builder.save()

	print('Done exporting!')

	#test_image=123
	#mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)

	#print restore_sess.run(y, feed_dict={x: [mnist_data.test.images[test_image]], keep_prob: 1.0})
