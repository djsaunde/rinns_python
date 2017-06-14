import os
import argparse
import numpy as np
from get_tiny_imagenet import get_data

import keras
from keras import losses
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
import sys 
import tensorflow as tf


def train_tiny_imagenet(hardware='cpu', batch_size=100, num_epochs=25, num_classes=200):
	# Load data
	x_train, y_train, x_val, y_val = load_tiny_imagenet(num_classes)
	print(x_train.shape)
	print(y_train.shape)
	print(x_val.shape)
	print(y_val.shape)
	
	if hardware == 'gpu':
		devices = ['/gpu:0']
	elif hardware == '2gpu':
		devices = ['/gpu:0', '/gpu:1']
	else:
		devices = ['/cpu:0']

	# Run on chosen processors
	for d in devices:
		with tf.device(d):
			model = Sequential()

			"""Block 1"""
			model.add(Conv2D(128, (3, 3), strides=(1,1), padding='same', 
					  input_shape=x_train.shape[1:]))
			model.add(BatchNormalization())
			model.add(Conv2D(128, (3, 3), strides=(1,1), padding='same'))
			model.add(BatchNormalization())
			model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
			model.add(Activation('relu'))
			
			"""Block 2"""
			model.add(Conv2D(128, (3, 3), strides=(1,1), padding='same'))
			model.add(BatchNormalization())
			model.add(Conv2D(128, (3, 3), strides=(1,1), padding='same'))
			model.add(BatchNormalization())
			model.add(Activation('relu'))
			model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

			"""Block 3"""
			model.add(Conv2D(128, (3, 3), strides=(1,1), padding='same'))
			model.add(BatchNormalization())
			model.add(Conv2D(128, (3, 3), strides=(1,1), padding='same'))
			model.add(BatchNormalization())
			model.add(Activation('relu'))
			model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

			"""Block 4"""
			model.add(Conv2D(128, (3, 3), strides=(1,1), padding='same'))
			model.add(BatchNormalization())
			model.add(Conv2D(128, (3, 3), strides=(1,1), padding='same'))
			model.add(BatchNormalization())
			model.add(Activation('relu'))
			model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
			
			
			"""Block 5"""
			model.add(Flatten())
			model.add(Dense(128))
			model.add(BatchNormalization())
			model.add(Activation('relu'))

			"""Output Layer"""
			model.add(Dense(num_classes))

			"""Loss Layer"""
			model.add(Activation('softmax'))

			"""Optimizer"""
			model.compile(loss=losses.categorical_crossentropy, 
						  optimizer='adam', metrics=['accuracy'])

	model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, 
			  validation_data=(x_val, y_val), shuffle=True)

def load_tiny_imagenet(num_classes=200):
	# Path to tiny imagenet dataset
	path = os.path.join('tiny-imagenet-200')
	# Generate data fields - test data has no labels so ignore it
	classes, x_train, y_train, x_val, y_val = get_data(path)
	# Get number of classes specified in order from [0, num_classes)
	if num_classes > 200:
		print('Set number of classes to maximum of 200\n')
		num_classes = 200
	elif num_classes != 200:
		train_indices = [index for index, label in enumerate(y_train) if label < num_classes]
		val_indices = [index for index, label in enumerate(y_val) if label < num_classes]
		x_train = x_train[train_indices]
		y_train = y_train[train_indices]
		x_val = x_val[val_indices]
		y_val = y_val[val_indices]
	
	# Format data to be the correct shape
	x_train = np.einsum('iljk->ijkl', x_train)
	x_val = np.einsum('iljk->ijkl', x_val)

	# Convert labels to one hot vectors
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_val = keras.utils.to_categorical(y_val, num_classes)

	return x_train, y_train, x_val, y_val	

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train a convolutional neural network on the Tiny-Imagenet dataset.')
	parser.add_argument('--hardware', type=str, default='cpu', help='cpu, gpu, or 2gpu currently supported.')
	parser.add_argument('--batch_size', type=int, default=100)
	parser.add_argument('--num_epochs', type=int, default=25)
	parser.add_argument('--num_classes', type=int, default=200)
	args = parser.parse_args()
	hardware, batch_size, num_epochs, num_classes = args.hardware, args.batch_size, args.num_epochs, args.num_classes

	# Possibly change num_classes to be a list of specific classes?
	train_tiny_imagenet(hardware, batch_size, num_epochs, num_classes)
