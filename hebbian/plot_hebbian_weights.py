'''
Visualize the weights of all Hebbian learning layers over the training procedure.
'''

__author__ = 'Dan Saunders'

import os
import sys
import keras
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from hebbian import Hebbian

# get command-line arguments from users
parser = argparse.ArgumentParser(description='Visualize the weights of all Hebbian learning layers over the training procedure.')
parser.add_argument('--model_name', default='mnist_hebbian', type=str, help='The name of the network model whose Hebbian layer weights we want to visualize.')
parser.add_argument('--lmbda', type=float, default=1.0, help='Hebbian learning layer lateral connection efficacy parameter.')
parser.add_argument('--eta', type=float, default=0.0005, help='Hebbian learning layer learning rate.')
parser.add_argument('--connectivity', type=str, default='all', help='Hebbian learning layer connectivity pattern.')
args = parser.parse_args()

model_name, lmbda, eta, connectivity = args.model_name, args.lmbda, args.eta, args.connectivity

weights_path = os.path.join('..', 'work', 'training', model_name, '_'.join([str(lmbda), str(eta), connectivity]))
if not os.path.isdir(weights_path):
	raise Exception('Weights do not exist for this model. Has it been trained?')

# load up weights file into memory
hebbian_weights = {}
for f in os.listdir(weights_path):
	# create empty list for this model checkpoint's Hebbian learning layer weights
	epoch = int(f.split('-')[1])
	hebbian_weights[epoch] = []

	# load up the associated model first
	model = keras.models.load_model(os.path.join(weights_path, f), custom_objects={'Hebbian' : Hebbian})
	
	# get only the weights of the Hebbian learning layers
	for layer in model.layers():
		if 'hebbian' in layer.name:
			hebbian_weights.append(layer.weights)

