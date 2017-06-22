'''
Implementation of a Hebbian learning layer to use as a computational module in building deep neural networks.
'''

__author__ = 'Dan Saunders'

from keras import backend as K
from keras.engine.topology import Layer

import numpy as np
import tensorflow as tf


class Hebbian(Layer):
	
	
	def __init__(self, output_dim, lmbda=1, eta=0.005, connectivity='all', **kwargs):
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
	

	def build(self, input_shape):
		# Create weight variable for this layer.
		self.kernel = self.add_weight(name='kernel', shape=(np.prod(input_shape[1:]), np.prod(self.output_dim)), initializer='uniform', trainable=False)
		super(Hebbian, self).build(input_shape)


	def call(self, x):
		x_shape = x.get_shape().as_list()
		batch_size = tf.shape(x)[0]
		new_x = tf.reshape(x, (batch_size, np.prod(x_shape[1:])))
		correlation = new_x[:, :, np.newaxis] * new_x[:, :, np.newaxis]
		activations = self.lmbda * K.dot(self.kernel, correlation)
		self.kernel = self.eta * correlation

		print(tf.shape(activations))
		print(self.kernel)
		print(x_shape)

		return K.reshape(activations, tf.shape(x))
