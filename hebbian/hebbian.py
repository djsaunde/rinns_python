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
	
	
	def __init__(self, output_dim, lmbda, eta, connectivity, **kwargs):
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
		self.eta = eta
		self.connectivity = connectivity

		super(Hebbian, self).__init__(**kwargs)
	
	
	def random_conn_init(self, shape, dtype=None):
		A = np.random.random(shape)
		A[A < 0.1] = 0
		return tf.constant(A, dtype=tf.float32)


	def build(self, input_shape):
		# Create weight variable for this layer.
		self.kernel = self.add_weight(name='kernel', shape=(np.prod(input_shape[1:]), np.prod(self.output_dim)), initializer='uniform', trainable=False)
		self.kernel = self.kernel * tf.diag(tf.zeros(self.output_dim))
		super(Hebbian, self).build(input_shape)


	def call(self, x):
		x_shape = tf.shape(x)
		batch_size = tf.shape(x)[0]

		# reshape to (batch_size, product of other dimensions) shape
		x = tf.reshape(x, (tf.reduce_prod(x_shape[1:]), batch_size))

		# compute activations using Hebbian-like update rule
		activations = x + self.lmbda * tf.matmul(self.kernel, x)

		# compute outer product of activations matrix with itself
		outer_product = tf.matmul(tf.expand_dims(x, 1), tf.expand_dims(x, 0))

		# update the weight matrix of this layer
		self.kernel = self.kernel + tf.multiply(self.eta, tf.reduce_mean(outer_product, axis=2))
		self.kernel = self.kernel * tf.diag(tf.zeros(self.output_dim))

		return K.reshape(activations, x_shape)


	def get_config(self):
		return dict(list(base_config.items()) + list({'lmbda' : self.lmbda, 'eta' : self.eta, 'connectivity' : self.connectivity}))
		# return cls(**config)
