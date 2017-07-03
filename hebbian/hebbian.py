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
	
	
	def __init__(self, output_dim, lmbda=1.0, eta=0.0005, connectivity='random', connectivity_prob=0.25, **kwargs):
		'''
		Constructor for the Hebbian learning layer.

		args:
			output_dim - The shape of the output / activations computed by the layer.
			lambda - A floating-point valued parameter governing the strength of the Hebbian learning activation.
			eta - A floating-point valued parameter governing the Hebbian learning rate.
			connectivity - A string which determines the way in which the neurons in this layer are connected to
				the neurons in the previous layer.
		'''
		self.output_dim = output_dim
		self.lmbda = lmbda
		self.eta = eta
		self.connectivity = connectivity
		self.connectivity_prob = connectivity_prob

		if self.connectivity == 'random':
			self.B = np.random.random(self.output_dim) < self.connectivity_prob
		elif self.connectivity == 'zero':
			self.B = np.zeros(self.output_dim)

		super(Hebbian, self).__init__(**kwargs)
	
	
	def random_conn_init(self, shape, dtype=None):
		A = np.random.normal(0, 1, shape)
		A[self.B] = 0
		return tf.constant(A, dtype=tf.float32)


	def zero_init(self, shape, dtype=None):
		return np.zeros(shape)


	def build(self, input_shape):
		# create weight variable for this layer according to user-specified initialization
		if self.connectivity == 'all':
			self.kernel = self.add_weight(name='kernel', shape=(np.prod(input_shape[1:]), \
								np.prod(self.output_dim)), initializer='uniform', trainable=False)
		elif self.connectivity == 'random':
			self.kernel = self.add_weight(name='kernel', shape=(np.prod(input_shape[1:]), \
								np.prod(self.output_dim)), initializer=self.random_conn_init, trainable=False)
		elif self.connectivity == 'zero':
			self.kernel = self.add_weight(name='kernel', shape=(np.prod(input_shape[1:]), \
								np.prod(self.output_dim)), initializer=self.zero_init, trainable=False)
		else:
			raise NotImplementedError
		
		# ensure that no neuron is laterally connected to itself
		self.kernel = self.kernel * tf.diag(tf.zeros(self.output_dim))

		# call superclass "build" function
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
		self.kernel = tf.multiply(self.kernel, self.B)
		self.kernel = self.kernel * tf.diag(tf.zeros(self.output_dim))

		return K.reshape(activations, x_shape)
