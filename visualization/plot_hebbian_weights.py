'''
Plot the evolution of the weights of all Hebbian learning layers in a given network over all training epochs.
'''

__author__ = 'Dan Saunders'

import os
import sys
import keras
import argparse
import numpy as np
import matplotlib.pyplot as plt

from hebbian import Hebbian


parser = argparse.ArgumentParser(description='Plot the evolution of the weights of all Hebbian learning layers in a given network over all training epochs.')
parser.add_argument('--model_name', type=str, default='mnist_hebbian', help='The name of the network model whose Hebbian layer weights we want to plot.')
args = parser.parse_args()

model_name = args.model_name

# get the relative path to this network's weights
weights_path = os.path.join('..', 'work', 'training', model_name)

# load weights from all hebbian layers
hebbian_weights = {}
for filename in os.listdir(weights_path):
	print('pizza')
	epoch = int(filename.split('-')[1])
	hebbian_weights[epoch] = []
	this_model = keras.models.load_model(os.path.join(weights_path, filename), custom_objects={'Hebbian' : Hebbian})
	print(this_model + 'hi')

print('hello')
