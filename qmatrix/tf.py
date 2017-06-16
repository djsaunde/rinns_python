# Mimics the functionality of "build_all_q.m" from Thomas Watson's Representations in Neural Networks project code.

__author__ = 'Dan Saunders'

import tensorflow as tf
import numpy as np
import argparse
import keras
import sys
import os

# ignore Tensorflow CPU compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
			
test = tf.global_variables_initializer()

def corr(x, y):
    """Correlate each n with each m.

    Parameters
    ----------
    x : np.array
      Shape N X T.

    y : np.array
      Shape M X T.

    Returns
    -------
    np.ar		q_min = tf.Variable(q_min)ray
      N X M array in which each element is a correlation coefficient.

    """
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must ' +
                         'have the same number of timepoints.')
    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)
    cov = np.dot(x,
                 y.T) - n * np.dot(mu_x[:, np.newaxis],
                                  mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


# parse optional arguments from user
parser = argparse.ArgumentParser(description='Compute the Q-matrices of all layers of a given neural network.')
parser.add_argument('--hardware', type=str, default='cpu', help='Use of cpu, gpu, or 2gpu currently supported.')
parser.add_argument('--model_name', type=str, default='cifar10_lenet', help='The name of the model whose Q-matrices we wish to compute.')
parser.add_argument('--best_criterion', type=str, default='val_loss', help='The criterion used to pick the "best" version of the network.')
parser.add_argument('--batch_size', type=int, default=100, help='Number of correlation thresholds to try.')
parser.add_argument('--use_weights', type=bool, default=False, help='Whether to use weighted Q-matrices.')
args = parser.parse_args()

hardware, model_name, best_criterion, batch_size, use_weights = args.hardware, args.model_name, args.best_criterion, args.batch_size, args.use_weights

qmatrix_path = os.path.join('..', 'work', 'qmatrix', model_name)
if not os.path.isdir(qmatrix_path):
    os.makedirs(qmatrix_path)

if hardware == 'gpu':
	d = '/gpu:0'
else:
	d = '/cpu:0'

# Handle gpu error with tf.Variable
config = tf.ConfigProto(allow_soft_placement=True)

with tf.device(d):
	with tf.Session(config=config) as sess:
		# load the network model from disk
		model_path = os.path.join('..', 'work', 'training', model_name, 'best_weights_' + best_criterion + '.hdf5')
		model = keras.models.load_model(model_path)
		print(model.summary())

		# get network metadata
		num_layers = len(model.layers)
		num_classes = model.layers[-1].output_shape[1]

		# get validation data labels
		activation_path = os.path.join('..', 'work', 'activations', model_name)
		labels = (np.load(os.path.join(activation_path, 'labels.npy')))

		# compute Q-matrices starting from output layer
		for layer_idx in range(num_layers, 0, -1):
			print('Computing Q-matrix for layer ' + str(layer_idx))
			
			# Step 1: Load old Q-matrix
			if layer_idx == num_layers:
				old_q = (np.identity(num_classes))
				old_q_size = ([num_classes, num_classes])
				old_col_weight = (np.ones(num_classes))
				old_class_map = ((range(num_classes)))
				threshold = 0
			else:
				# qfname = 'q' + str(layer_idx+1) + '_temp.npy'
				old_q = (np.load(os.path.join('q_out', qfname)))
				# These could probably be changed to not use np.copy
				old_col_weight = (np.copy(col_weight)) 
				old_q_size = (np.copy(q_size))
				old_class_map = (np.copy(class_map))
			
			# the maximum number of columns the new Q-matrix can have is the number of
			# neurons in the previous Q-matrix times the number of classes
			
			# np.unique might not work on tensor
			max_cols = old_q.shape[0] * len(np.unique(old_class_map))

			neuron_map = np.zeros(max_cols)
			class_map = np.zeros(max_cols)
			col_weight = np.zeros(max_cols)

			# iterate through each neuron in the previous layer
			other_idx = 0
			print((old_q_size))
			for neuron_idx in range(old_q_size[0]):
				# select each row with this previous neuron in the previous Q-matrix
				row_select = old_q[neuron_idx, :] > threshold
				row_num = tf.reduce_sum(row_select)

				# if there aren't any, simply move on
				if row_num == 0:
					continue
				
				# map these neurons to classes
				old_col_weight = (np.ones(num_classes))
				old_class_map = ((range(num_classes)))

				class_select = old_class_map[row_select]
				weight_select = old_col_weight[row_select]

				# we don't want multiple rows of the same class, since they would have the same data.
				# instead, we note how many would be there so we can weight appropriately during classifcation.
				classes = (np.unique(class_select))
				row_num = len(classes)
				r = ([ tf.reduce_sum(weight_select[class_select == cls]) for cls in classes ])

				# select the correlations for this neuron and each class that it's associated with
				neuron_map[other_idx:other_idx + row_num] = neuron_idx
				class_map[other_idx:other_idx + row_num] = classes
				col_weight[other_idx:other_idx + row_num] = r;

				other_idx += row_num
			
			# remove unused columns
			neuron_map = (np.delete(neuron_map, np.s_[other_idx:max_cols]))
			class_map = (np.delete(class_map, np.s_[other_idx:max_cols]))
			col_weight = (np.delete(col_weight, np.s_[other_idx:max_cols]))

			del old_q, old_class_map, old_col_weight

			# Step 2: Build new Q-matrix
			qfname_temp = 'q' + str(layer_idx) + '_temp.npy'
			qfname = 'q' + str(layer_idx) + '.npy'

			# load activation data for the two layers we're correlating
			if 'data_a_map' in locals():
				data_b = data_a # we have already loaded it from the last iteration
			else:
				data_b = np.load(os.path.join(activation_path, 'layer' + str(layer_idx) + '.npy')).T
				data_b = data_b.reshape((np.prod(data_b.shape[:-1]), data_b.shape[-1]))
				data_b = (data_b)
			
			data_a = np.load(os.path.join(activation_path, 'layer' + str(layer_idx - 1) + '.npy')).T
			data_a = data_a.reshape((np.prod(data_a.shape[:-1]), data_a.shape[-1]))
			data_a = (data_a)

			q_size = [np.prod(data_a.shape[:-1]), len(neuron_map)]

			print(data_a.shape, data_b.shape)

			print('Building Q' + str(layer_idx) + ' (' + str(q_size[0]) + 'x' + str(q_size[1]) + ')')

			# we will compute maximum and minimum as we go along
			q_max = -np.inf
			q_min = np.inf

			old_class_map = (np.copy(class_map))
			old_col_weight = (np.copy(col_weight))
			old_neuron_map = (np.copy(neuron_map))

			# go through each class (correlation matrix depends on class)
			other_idx = 0
			for label_idx in range(int(np.max(labels) + 1)):
				# select columns in this class
				select = old_class_map == label_idx
				row_num = np.sum(select)

				# if there are no neurons with this label
				if row_num == 0:
					continue

				# correlate the neurons based on images in this class
				corr_data = corr(data_a[:, labels == label_idx], data_b[:, labels == label_idx])

				# select the correlations of the neurons that these columns came from
				corr_data = corr_data[:, np.asarray(old_neuron_map[select], dtype=np.int)]

				# update minimum, maximum
				q_max = np.maximum(q_max, np.max(corr_data))
				#q_max = tf.Variable(q_max)

				q_min = np.minimum(q_min, np.min(corr_data))
				#q_min = tf.Variable(q_min)

				# this reorders the maps, so we have to update them
				class_map[other_idx:other_idx + row_num] = old_class_map[select]
				col_weight[other_idx:other_idx + row_num] = old_col_weight[select]
				neuron_map[other_idx:other_idx + row_num] = old_neuron_map[select]

				if not 'to_write' in locals():
					to_write = corr_data
				else:
					to_write = np.hstack((to_write, corr_data))

				print(to_write.shape)

				other_idx += row_num
			
			# write correlation data to Q-matrix file
			np.save(qfname_temp, to_write)
			
			del corr_data, to_write
			del data_b, old_class_map, old_col_weight

			if layer_idx == 1:
				del data_a
			
			# Step 3: normalize and multiply by weights
			layer = model.layers[layer_idx-1]
			has_weights = 'dense' in layer.name or 'conv' in layer.name
			
			if has_weights and use_weights:
				# we only have to normalize and multiply by weights if the layer has them
				# we compute the "unshared" weights by convolving the identity matrix of size
				# equal to the number of neurons. This means each output image is the weights
				# that make up one input neuron. The problem is that they need to be applied
				# to the transposes Q-matrix, so we write them to disk.
				w_size = layer.get_weights().shape
				weights = layer.get_weights()

				im_size = np.prod(layer.input_shape[1:])
			else:
				if not os.path.isdir('q_out'):
					os.makedirs('q_out')
				
				qfile = np.load(qfname_temp)
				np.save(os.path.join('q_out', qfname), qfile)
