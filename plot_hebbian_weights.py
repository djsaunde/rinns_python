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


# get command-line arguments from users
parser = argparse.ArgumentParser(description='Visualize the weights of all Hebbian learning layers over the training procedure.')
parser.add_argument('--model_name', default='mnist_hebbian', type=str, help='The name of the network model whose Hebbian layer weights we want to visualize.')
args = parser.parse_args()

model_name = args.model_name

weights_path = os.path.join('..', 'work', 'training', model_name)
if not os.path.isdir(weights_path):
	os.makedirs(weights_path)

# load up weights file into memory
hebbian_weights = {}
for f in os.listdir(weights_path):
	# create empty list for this model checkpoint's Hebbian learning layer weights
	epoch = int(f.split('-')[1])
	hebbian_weights[epoch] = []

	# load up the associated model first
	model = keras.models.load(f)
	
	# get only the weights of the Hebbian learning layers
	for layer in model.layers():
		if 'hebbian' in layer.name:
			hebbian_weights.append(layer.weights)
