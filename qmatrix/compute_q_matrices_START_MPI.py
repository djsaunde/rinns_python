# Mimics the functionality of "build_all_q.m" from Thomas Watson's Representations in Neural Networks project code.

import pycuda.gpuarray as gpuarray
import tensorflow as tf
import numpy as np
import argparse
import keras
import math
import sys
import os

np.set_printoptions(threshold=np.nan)

# ignore Tensorflow CPU compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

sess = tf.Session()


def corr(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);

    # Finally get corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))


def classify(qs, layer_idx, netpath, mem, range):
	result = None # TODO

	# get the Q-matrix
	q = np.load(qs['qfname'])

	# get layer activations and labels
	activations = np.load(os.path.join(qs['activation_path'], 'layer' + str(layer_idx) + '.npy'))
	activations = activations.reshape((np.prod(activations.shape[:-1]), activations.shape[-1]))
	labels = np.load(os.path.join(qs['activation_path'], 'layer' + str(layer_idx) + '.npy'))
	num_classes = labels.max()

	# this contains one row for each class and one column for each image
	classification = np.zeros((num_classes, activations.shape[1]))

	# we want to keep the amount of activation data in GPU memory under mem.activations bytes
	imstep = math.floor((memory.activations / 4) / activations.shape[0])
	if imstep >= activations.shape[1]:
		gpu_activations = gpuarray.to_gpu(activations)		

	col_weight = gpuarray.to_gpu(qs['col_weight'])
	for row_idx in range(qs['range']):
		print('Classifying using Q' + str(layer_idx) + ' and threshold ' + qs['range'][row_idx] + ' over ' + str(activations.shape[0]) + ' neurons.')
		
		for image in range(0, activations.shape[1], imstep):
			# number of images to do in this loop
			this_im_end = min(image + im_step, activations.shape[1])
			num_ims = this_im_end - im

			# upload the data to the GPU
			if imstep < activations.shape[1]:
				gpu_activations = gpuarray.to_gpu(activations[:, im:this_im_end])
			
			# multiply every column in the Q-matrix by a column of activation (one for each image)
			# sum up all rows so that it's one row high
			# each column is associated with a class
			# sum the columns associated with each class
			# maximum value is the classification
			for cls in range(num_classes):
				# select columns of the Q-matrix which have this class
				select = qs['class_map'] == cls

				if np.sum(select) == 0:
					continue

				# we want to keep the amount of resulting data under memory.q bytes
				# we need to figure out how many columns of the Q-matrix we should
				# put in GPU memory after each iteration
				col_step = math.floor((memory.q / 4) / q.shape[0])

				if col_step == 0:
					raise Exception('col_step = 0')

				# calculate loop indices
				col_vals = range(0, q.shape[1], col_step)

				# we will store a result for each iteration of the loop
				tr = np.zeros((num_ims, len(col_vals)))

				tr_idx = 0
				for col in col_vals:
					this_end = min(col + col_step, len(select))
					sub_select = select(col, this_end)
					sub_q = gpuarray.to_gpu(q[:, sub_select] > range(row_idx))
					tr[:, tr_idx] = (gpu_activations * sub_q) * col_weight[sub_select]
					
					del sub_q

					tr_idx += 1

				classification[cls, im:this_im_end] = comm.gather(np.sum(tr, axis=1))
				del tr

			if imstep < activations.shape[1]:
				del gpu_activations


# parse optional arguments from user
parser = argparse.ArgumentParser(description='Compute the Q-matrices of all layers of a given neural network.')
parser.add_argument('--hardware', type=str, default='cpu', help='Use of cpu, gpu, or 2gpu currently supported.')
parser.add_argument('--model_name', type=str, default='cifar10_lenet', help='The name of the model whose Q-matrices we wish to compute.')
parser.add_argument('--best_criterion', type=str, default='val_loss', help='The criterion used to pick the "best" version of the network.')
parser.add_argument('--batch_size', type=int, default=100, help='Number of correlation thresholds to try.')
parser.add_argument('--use_weights', type=str, default='False', help='Whether to use weighted Q-matrices.')
args = parser.parse_args()

hardware, model_name, best_criterion, batch_size, use_weights = args.hardware, args.model_name, args.best_criterion, args.batch_size, args.use_weights

if use_weights.lower() == 'true':
	use_weights = True
elif use_weights.lower() == 'false':
	use_weights = False
else:
	raise Exception('Enter True or False (case insensitive)!')

qmatrix_path = os.path.join('..', 'work', 'qmatrix', model_name)
if not os.path.isdir(qmatrix_path):
    os.makedirs(qmatrix_path)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
comm_size = comm.Get_size()

# load the network model from disk
model_path = os.path.join('..', 'work', 'training', model_name, 'best_weights_' + best_criterion + '.hdf5')
model = keras.models.load_model(model_path)

############################################
if rank == 0:
	startqi = len(model.layers)
	machine_names = np.array(range(comm_size))
	for idx in range(1, comm_size):
		mrank = []
		while len(mrank) == 0:
			# Not sure what source and tag are
			# status = MPI.status() - optional 3rd argument - default=None
			mrank = comm.Probe(MPI.ANY_SOURCE, 5)
		for arank in mrank:
			# Matlab MPI_Recv(source, tag, comm)
			# comm.recv(source=0, tag=0, status=None)
			machine_names[arank] = comm.recv(arank, 5)
	# Broadcast completed list
	# comm.bcast(obj=None, int root=0)
	# no clue if this is correct
	comm.bcast(machine_names, root=0)
else:
	# Send our name (Machine name is passed)
	# also probably wrong
	comm.send(machine_name)
	# Get everyone else's names
	machine_names = comm.recv(0, 6)
#############################################

memory = {}
# This doesn't give the correct amount of available gpu memory, probably has to do with 'context
#memory['total'] = pycuda.driver.mem_get_info()[0]
#print(pycuda.driver.mem_get_info())
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)
memory['total'] = nvmlDeviceGetMemoryInfo(handle).free
print(memory['total'])
nvmlShutdown()

if memory['total'] < 2 ** 30:
	raise Exception('Less than 1Gb GPU memory available.')

memory['reserved'] = memory['total'] * 0.85
memory['q'] = memory['reserved'] * 0.5
memory['activations'] = memory['reserved'] * 0.5

if rank == 0:
	print(model.summary())

	# get network metadata
	num_layers = len(model.layers)
	num_classes = model.layers[-1].output_shape[1]

	# get validation data labels
	activation_path = os.path.join('..', 'work', 'activations', model_name)
	labels = np.load(os.path.join(activation_path, 'labels.npy'))

	# compute Q-matrices starting from output layer
	for layer_idx in range(num_layers, 1, -1):
		print('Computing Q-matrix for layer ' + str(layer_idx))

		if 'q' in locals():
			del q

		# Step 1: Load old Q-matrix
		if layer_idx == num_layers:
			old_q = np.identity(num_classes)
			old_q_size = np.array([num_classes, num_classes])
			old_col_weight = np.ones(num_classes)
			old_class_map = np.array(range(num_classes))
			threshold = 0
		else:
			# Dan had qfname commented out
			qfname = 'q' + str(layer_idx + 1) + '_temp.npy'
			old_q = np.load(os.path.join('q_out', qfname))
			old_col_weight = np.copy(col_weight)
			old_q_size = np.copy(q_size)
			old_class_map = np.copy(class_map)
	
		# the maximum number of columns the new Q-matrix can have is the number of
		# neurons in the previous Q-matrix times the number of classes
		max_cols = old_q.shape[0] * len(np.unique(old_class_map))

		neuron_map = np.zeros(max_cols)
		class_map = np.zeros(max_cols)
		col_weight = np.zeros(max_cols)

		# iterate through each neuron in the previous layer
		other_idx = 0
		for neuron_idx in range(old_q_size[0]):
			# select each row with this previous neuron in the previous Q-matrix
			row_select = old_q[neuron_idx, :] > threshold
			row_num = np.sum(row_select)

			# if there aren't any, simply move on
			if row_num == 0:
				continue
		
			# map these neurons to classes
			class_select = old_class_map[row_select]
			weight_select = old_col_weight[row_select]

			# we don't want multiple rows of the same class, since they would have the same data.
			# instead, we note how many would be there so we can weight appropriately during classifcation.
			classes = np.unique(class_select)
			row_num = len(classes)
			r = np.array([ np.sum(weight_select[class_select == cls]) for cls in classes ])

			# select the correlations for this neuron and each class that it's associated with
			neuron_map[other_idx:other_idx + row_num] = neuron_idx
			class_map[other_idx:other_idx + row_num] = classes
			col_weight[other_idx:other_idx + row_num] = r;

			other_idx += row_num
	
		# remove unused columns
		neuron_map = np.delete(neuron_map, np.s_[other_idx:max_cols])
		class_map = np.delete(class_map, np.s_[other_idx:max_cols])
		col_weight = np.delete(col_weight, np.s_[other_idx:max_cols])

		del old_q, old_class_map, old_col_weight

		# Step 2: Build new Q-matrix
		qfname_temp = 'q' + str(layer_idx) + '_temp.npy'
		qfname = 'q' + str(layer_idx) + '.npy'

		# load activation data for the two layers we're correlating
		if 'data_a_map' in locals():
			data_b = np.copy(data_a) # we have already loaded it from the last iteration
		else:
			data_b = np.load(os.path.join(activation_path, 'layer' + str(layer_idx) + '.npy')).T
			data_b = data_b.reshape((np.prod(data_b.shape[:-1]), data_b.shape[-1]))
	
		data_a = np.load(os.path.join(activation_path, 'layer' + str(layer_idx - 1) + '.npy')).T
		data_a = data_a.reshape((np.prod(data_a.shape[:-1]), data_a.shape[-1]))

		q_size = [np.prod(data_a.shape[:-1]), len(neuron_map)]

		print('Building Q' + str(layer_idx) + ' (' + str(q_size[0]) + 'x' + str(q_size[1]) + ')')

		# we will compute maximum and minimum as we go along
		q_max = -np.inf
		q_min = np.inf

		old_class_map = np.copy(class_map)
		old_col_weight = np.copy(col_weight)
		old_neuron_map = np.copy(neuron_map)

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
			q_min = np.minimum(q_min, np.min(corr_data))

			# this reorders the maps, so we have to update them
			class_map[other_idx:other_idx + row_num] = old_class_map[select]
			col_weight[other_idx:other_idx + row_num] = old_col_weight[select]
			neuron_map[other_idx:other_idx + row_num] = old_neuron_map[select]

			if not 'q' in locals():
				q = corr_data
			else:
				q = np.hstack((q, corr_data))

			other_idx += row_num

		# write correlation data to Q-matrix file
		np.save(os.path.join('q_out', qfname_temp), q)

		del corr_data, q
		del data_b, old_class_map, old_col_weight

		if layer_idx == 1:
			del data_a
	
		# Step 3: normalize and multiply by weights
		layer = model.layers[layer_idx - 1]
		has_weights = 'dense' in layer.name or 'conv' in layer.name

		if has_weights and use_weights:
			# we only have to normalize and multiply by weights if the layer has them
			# we compute the "unshared" weights by convolving the identity matrix of size
			# equal to the number of neurons. This means each output image is the weights
			# that make up one input neuron. The problem is that they need to be applied
			# to the transposes Q-matrix, so we write them to disk.
			w_size = layer.input_shape

			weights = layer.get_weights()[0]

			im_size = np.prod(w_size[1:])
			imstep = max(1, math.floor(((2**30)/4) / im_size))

			for im in range(0, im_size, imstep):
				this_end = min(im + imstep - 1, im_size)
				this_step = this_end - im
			
				im_data = np.zeros((im_size, this_step))
			
				for idx in range(im, this_end):
					im_data[idx, idx - im] = 1;

				if 'conv' in layer.name:
					im_data = im_data.reshape([w_size[1], w_size[2], w_size[3], this_step])
			
					im_data, weights = np.einsum('ijkl->lijk', im_data), weights # np.einsum('ijkl->iklj', weights)
			
				im_data = im_data.astype(np.float32)

				if 'dense' in layer.name:
					im_data = tf.expand_dims(im_data, 0)
					weights = tf.expand_dims(weights, 0)
					output = tf.map_fn(
						lambda inputs : tf.nn.xw_plus_b(
							inputs[0], # tf.expand_dims(tf.transpose(inputs[0]), 0), # inputs[0],
							inputs[1], # tf.expand_dims(inputs[1], -1),
							np.zeros(inputs[1].shape[1]).astype(np.float32)
						),
						elems=[im_data, weights],
						dtype=tf.float32
					)
				
					result = output.eval(session=sess).reshape((weights.shape[2], weights.shape[1]))

				if 'conv' in layer.name:
					output = tf.nn.conv2d(im_data, weights, strides=[1,1,1,1], padding='SAME')	
					result = output.eval(session=sess).reshape((np.prod(output.shape[1:]), output.shape[0]))

				if 'to_write' not in locals():
					to_write = result
				else:
					to_write = np.hstack((to_write, result))

			np.save(os.path.join('q_out', 'w' + str(layer_idx) + '.npy'), to_write)
			del to_write

			old_col_weight = col_weight
			old_class_map = class_map

			old_q_max = q_max
			old_q_min = q_min
			q_max = -np.inf
			q_min = np.inf

			weights = np.load(os.path.join('q_out', 'w' + str(layer_idx) + '.npy'))
			q = np.load(os.path.join('q_out', qfname_temp))

			other_idx = 0
			for neuron_idx in range(old_q_size[0]):
				# select columns of the Q matrix with this neuron
				select = neuron_map == neuron_idx
				row_num = sum(select)
			
				if row_num == 0:
					continue

				# get these columns and normalize
				result = q[:, select] - old_q_min
				result = result / (old_q_max - old_q_min)

				# multiply by appropriate weights
				result = result * np.expand_dims(weights[neuron_idx, :], axis=-1)

				# recompute max and min
				q_max = max([q_max, np.max(result)])
				q_min = min([q_min, np.min(result)])

				# write it to disk
				col_weight[other_idx:other_idx + row_num] = old_col_weight[select]
				class_map[other_idx:other_idx + row_num] = old_class_map[select]
			
				if 'q_new' not in locals():
					q_new = result
				else:
					q_new = np.hstack((q_new, result))
			
				del result
				other_idx = other_idx + row_num

			np.save(os.path.join('q_out', qfname), q_new)
		
			del q_new
		
		else:
			if not os.path.isdir('q_out'):
				os.makedirs('q_out')
		
			qfile = np.load(os.path.join('q_out', qfname_temp))
			np.save(os.path.join('q_out', qfname), qfile)

		# build information dictionary for this Q-matrix
		qs = {}
		qs['range'] = np.linspace(q_min, q_max, num_thresholds)
		qs['qfname'] = qfname
		qs['class_map'] = class_map
		qs['col_weight'] = col_weight
		qs['size'] = q_size
		qs['activation_path'] = activation_path

		print('\nBroadcasting resulting Q (' + str(qs['size'][0]) + 'x' + str(qs['size'][1]) + ')')

		'''
		# tell the other ranks about the Q-matrices
		comm.bcast(qs, root=0)

		# work on our subset of the data
		r = classify(qs, layer_idx, )
		'''

# rank is nonzero	
else:
	pass	

