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
		# get shape of input
		x_shape = tf.shape(x)

		# reshape input to be flat along non-batch dimensions
		x = tf.reshape(x, (x_shape[0], np.prod(x.get_shape().as_list()[1:])))
		
		# correlation = tf.reduce_mean(x[:, :, np.newaxis] * x[:, np.newaxis, :], axis=0)
		
		# calculate output based on current weights and input activations
		y = self.lmbda * tf.multiply(self.kernel, tf.reduce_mean(tf.multiply(x, tf.transpose(x)), axis=0))
		
		# update weights based on activations (
		corr = tf.reduce_mean(x[:, :, np.newaxis] * x[:, np.newaxis, :], axis=1)
		self.kernel = self.kernel + self.eta * corr

		return K.reshape(y, x_shape)

