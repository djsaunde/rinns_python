# Mimics the functionality of the +train.compute_activations script found in Thomas Watson's Representations in Neural Networks project code.
# @author Dan Saunders

import os
import keras
import argparse
import numpy as np
import pickle as p

from keras import backend
from keras.datasets import cifar10
from keras.models import load_model
from keras.utils import to_categorical

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# get command-line arguments from user
parser = argparse.ArgumentParser(description='Compute the layer-wise activations of a neural network over a given dataset.')
parser.add_argument('--model_name', type=str, default='cifar10_lenet', help='The name of the neural network model whose activations we want to compute and store.')
parser.add_argument('--dataset', type=str, default='cifar10', help='The dataset which we wish to computer network activations over.')
parser.add_argument('--best_criterion', type=str, default='val_loss', help='The criterion used to determine which network performed best over all epochs.')
args = parser.parse_args()

model_name, dataset, best_criterion = args.model_name, args.dataset, args.best_criterion

activations_path = os.path.join('..', 'work', 'activations', model_name)
if not os.path.isdir(activations_path):
	os.makedirs(activations_path)

# load data (validation data only for this step!)
print('...Loading validation data.')
if dataset == 'cifar10':
	(x_train, y_train), (x_valid, y_valid) = cifar10.load_data()
	# save validation labels to disk
	np.save(os.path.join(activations_path, 'labels.npy'), y_valid.reshape(10000))
	y_valid = to_categorical(y_valid, 10)
else:
	raise NotImplementedError

# load model
print('...Loading neural network model.')
model_path = os.path.join('..', 'work', 'training', model_name, 'best_weights_' + best_criterion + '.hdf5')
model = load_model(model_path)

# create functor for computing activations
inpt, outputs = model.input, [ layer.output for layer in model.layers ]
functor = backend.function([inpt] + [backend.learning_phase()], outputs)

# compute layer-wise activations
print('...Computing layer-wise activations over the validation data.')
activations = functor([x_valid, 1.])

# save activations to disk, making directories as needed
print('...Writing computed activations to file.')
if not os.path.isdir(activations_path):
	os.makedirs(activations_path)

for idx, layer_activation in enumerate(activations):
	np.save(os.path.join(activations_path, 'layer' + str(idx + 1) + '.npy'), layer_activation)

