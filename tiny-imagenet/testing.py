import os
import numpy
from load_tiny_imagenet import load_tiny_imagenet

# Path to tiny imagenet dataset
path = os.path.join('tiny-imagenet-200')
# Generate data fields from loading scripts
classes, x_train, y_train, x_val, y_val, x_test, y_test = load_tiny_imagenet(path)

# Test data has no labels, so shit everything down by 10000
x_test = x_val[:,:,:,:]
y_test = y_val[:]

x_val = x_train[90000:100000, :, :, :]
y_val = y_train[90000:100000]

x_train = x_train[0:90000, :, :, :]
y_train = y_train[0:90000]


