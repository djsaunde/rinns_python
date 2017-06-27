'''
Implementation of a Hebbian learning layer to use as a computational module in building deep neural networks.
'''

__author__ = 'Dan Saunders'

from keras import backend as K
from keras.engine.topology import Layer

import numpy as np
import tensorflow as tf

np.set_printoptions(threshold=np.nan)

sess = tf.Session()


class Hebbian(Layer):
	
	
	def __init__(self, output_dim, lmbda=1, eta=1.0, connectivity='random', **kwargs):
		'''
		Constructor for the Hebbian learning layer.

		args:
			output_dim - The shape of the output / activations computed by the layer.
			kwargs['lambda'] - A floating-point valued parameter governed the strength of the Hebbian learning activation.
			kwargs['connectivity'] - A string which determines the way in which the neurons in this layer are connected to
				the neurons in the previous layer.
		'''
		self.output_dim = output_dim
		self.lmbda = lmbda
		self.connectivity = connectivity
		self.eta = eta

		super(Hebbian, self).__init__(**kwargs)
	
	
	def random_conn_init(self, shape, dtype=None):
		print(shape)
		A = np.random.random(shape)
		A[A < 0.1] = 0
		A = tf.constant(A)

		return tf.cast(A, tf.float32)

		idx = tf.where(tf.not_equal(A, 0))
		sparse = tf.SparseTensor(idx, tf.gather_nd(A, idx), A.get_shape())
		return sparse


	def build(self, input_shape):
		# Create weight variable for this layer.
		self.kernel = self.add_weight(name='kernel', shape=(np.prod(input_shape[1:]), np.prod(self.output_dim)), initializer='uniform', trainable=False)
		super(Hebbian, self).build(input_shape)


	def call(self, x):
		x_shape = tf.shape(x)
		batch_size = tf.shape(x)[0]

		# reshape to (batch_size, product of other dimensions) shape
		x = tf.reshape(x, (tf.reduce_prod(x_shape[1:]), batch_size))

		# compute activations using Hebbian-like update rule
		activations = self.lmbda * tf.matmul(self.kernel, x)

		# sum contributions over each batch
		batch_sum = np.divide(tf.reduce_sum(x, axis=1), tf.cast(batch_size, tf.float32))

		# compute outer product of activations matrix with itself
		outer_product = tf.matmul(batch_sum[:, np.newaxis], batch_sum[np.newaxis, :])	

		# update the weight matrix of this layer
		self.kernel = self.kernel + tf.multiply(self.eta, tf.matmul(self.kernel, outer_product))

		# return properly-shaped computed activations
		return K.reshape(activations, x_shape)

