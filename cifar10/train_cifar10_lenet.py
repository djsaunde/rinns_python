# @author Ryan McCormick, Dan Saunders
# Slightly modified code modeled after: 
# https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py

import keras
import sys, os
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import losses
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.callbacks import ModelCheckpoint

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

model_name = 'cifar10_lenet'

train_path = os.path.join('..', 'work', 'training', model_name)
plots_path = os.path.join('..', 'plots', model_name)

if not os.path.isdir(train_path):
	os.makedirs(train_path)
if not os.path.isdir(plots_path):
	os.makedirs(plots_path)

parser = argparse.ArgumentParser(description='Train a convolutional neural network on the CIFAR-10 dataset.')
parser.add_argument('--hardware', type=str, default='cpu', help='Use of cpu, gpu, or 2gpu currently supported.')
parser.add_argument('--batch_size', type=int, default=100, help='Number of training / validation examples per minibatch.')
parser.add_argument('--num_epochs', type=int, default=25, help='Number of epochs (passes through training dataset) for which to train the network.')
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

		(x_train, y_train), (x_test, y_test) = cifar10.load_data()

		# Checking data sizes
		print('x_train shape:', x_train.shape)
		print(x_train.shape[0], 'train samples')
		print(x_test.shape[0], 'test samples')

		# Convert class vectors to binary class matrices
		y_train = keras.utils.to_categorical(y_train, 10)
		y_test = keras.utils.to_categorical(y_test, 10)

		# Feed-forward
		model = Sequential()

		"""Block 1"""
		# Filters(32), Slider_size(5,5), input_shape(32,32,3)
		model.add(Conv2D(32, (5, 5), strides=(1,1), padding='same', input_shape=x_train.shape[1:]))
		model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
		model.add(Activation('relu'))

		"""Block 2"""
		model.add(Conv2D(32, (5, 5), strides=(1,1), padding='same'))
		model.add(Activation('relu'))
		model.add(AveragePooling2D(pool_size=(3, 3), strides=(2,2), padding='same'))

		"""Block 3"""
		model.add(Conv2D(64, (5, 5), strides=(1,1), padding='same'))
		model.add(Activation('relu'))
		model.add(AveragePooling2D(pool_size=(3, 3), strides=(2,2), padding='same'))

		"""Block 4"""
		model.add(Flatten())
		model.add(Dense(64))
		model.add(Activation('relu'))

		"""Block 5"""
		model.add(Dense(10)) 

		"""Loss Layer"""
		model.add(Activation('softmax'))

		"""Optimizer"""
		model.compile(loss=losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

		x_train = x_train.astype('float32')
		x_test = x_test.astype('float32')
		x_train /= 255
		x_test /= 255

# check model checkpointing callback which saves only the "best" network according to the 'best_criterion' optional argument (defaults to validation loss)
model_checkpoint = ModelCheckpoint(os.path.join(train_path, 'weights-{epoch:02d}-' + best_criterion + '.hdf5'), monitor=best_criterion, save_best_only=False)

# fit the model using the defined 'model_checkpoint' callback
history = model.fit(x_train, y_train, batch_size, epochs=num_epochs, validation_data=(x_test, y_test), shuffle=True, callbacks=[model_checkpoint])

fig, axes = plt.subplots(2, sharex=True)
axes[0].plot(range(1, num_epochs + 1), history.history['acc'], '*:')
axes[0].plot(range(1, num_epochs + 1), history.history['val_acc'], '*:')
axes[0].set_ylabel('accuracy')
axes[1].set_xlabel('epoch')
axes[1].set_xticks(range(1, num_epochs + 1)) 
axes[0].legend(['train', 'validation'], loc='lower right')
axes[0].set_title(model_name + ' accuracy and loss')
axes[1].plot(range(1, num_epochs + 1), history.history['loss'], '*:')
axes[1].plot(range(1, num_epochs + 1), history.history['val_loss'], '*:')
axes[1].set_ylabel('loss')
axes[1].set_xlabel('epoch')
axes[1].set_xticks(range(1, num_epochs + 1)) 
axes[1].legend(['train', 'validation'], loc='upper right')

fig.savefig(os.path.join(plots_path, '%d_accuracy_loss.png' % (num_epochs)))

