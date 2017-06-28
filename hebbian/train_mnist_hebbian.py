'''
Implement a convolutonal neural network with Hebbian learning layers
which learns to classify the MNIST handwritten digit dataset.
'''

__author__ = 'Dan Saunders'

import keras
import sys, os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import losses
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint

from hebbian import Hebbian

# Suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

train_path = os.path.join('..', 'work', 'training', 'mnist_hebbian')
plots_path = os.path.join('..', 'plots')

if not os.path.isdir(train_path):
	os.makedirs(train_path)
if not os.path.isdir(plots_path):
	os.makedirs(plots_pat)

parser = argparse.ArgumentParser(description='Train a convolutional neural network on the CIFAR-10 dataset.')
parser.add_argument('--hardware', type=str, default='cpu', help='Use of cpu, gpu, or 2gpu currently supported.')
parser.add_argument('--batch_size', type=int, default=128, help='Number of training / validation examples per minibatch.')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs (passes through training dataset) for which to train the network.')
parser.add_argument('--best_criterion', type=str, default='val_loss', help='Criterion to consider when choosing the "best" model. Can also \
																use "val_acc", "train_loss", or "train_acc" (and perhaps others?).')
args = parser.parse_args()

hardware, batch_size, num_epochs, best_criterion = args.hardware, args.batch_size, args.num_epochs, args.best_criterion

### Pick CPU or GPU ###
if hardware == "cpu":
	device_names = ["/cpu:0"]
elif hardware == "gpu":
    device_names = ["/gpu:0"]
elif hardware == "2gpu":
	device_names = ["/gpu:0", "/gpu:1"]
else:
    raise NotImplementedError

### Run code on chosen devices ###
for d in device_names:
	with tf.device(d):
		num_classes = 10

		# mnist.load_data() returns 2 tuples split into training/testing
		(x_train, y_train), (x_valid, y_valid) = mnist.load_data()

		# If color channels are last parameter
		# Image dimensions are 28x28
		x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')[:1000, :, :, :]
		x_valid = x_valid.reshape(x_valid.shape[0], 28, 28, 1).astype('float32')[:100, :, :, :]

		# Normalize pixel values between 0 and 1 per channel
		x_train /= 255
		x_valid /= 255

		# Convert class label values to one-hot vectors
		y_train = keras.utils.to_categorical(y_train, num_classes)[:1000]
		y_valid = keras.utils.to_categorical(y_valid, num_classes)[:100]

		# Print out sample sizes
		print("Training samples:", x_train.shape[0])
		print("Test samples:", x_valid.shape[0])

		# Build model
		model = Sequential()
		model.add(Conv2D(32, (3, 3), activation='relu', input_shape=x_train.shape[1:]))
		# model.add(Hebbian(model.layers[-1].output_shape[1:]))
		model.add(Flatten())
		model.add(Dense(128, activation='relu'))
		model.add(Hebbian(128))
		model.add(Dense(64, activation='relu'))

		# Output layer
		model.add(Dense(num_classes, activation='softmax'))

		# Choosing loss function and optimizer
		model.compile(loss=keras.losses.categorical_crossentropy,
					  optimizer=keras.optimizers.Adam(),
					  metrics=['accuracy'])

# check model checkpointing callback which saves only the "best" network according to the 'best_criterion' optional argument (defaults to validation loss)
model_checkpoint = ModelCheckpoint(os.path.join(train_path, 'best_weights_' + best_criterion + '.hdf5'), monitor=best_criterion, save_best_only=True)

# fit the model using the defined 'model_checkpoint' callback
history = model.fit(x_train, y_train, batch_size, epochs=num_epochs, validation_data=(x_valid, y_valid), shuffle=True, callbacks=[model_checkpoint])

# plot training and validation loss and accuracy curves
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('iteration')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig(os.path.join(plots_path, 'mnist_hebbian_accuracy.png'))

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('iteration')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig(os.path.join(plots_path, 'mnist_hebbian_loss.png'))

