# Adoption of the save_best.m from Thomas Watson's Representations in Neural Networks projects
# Written by Dan Saunders

import tensorflow as tf
import argparse
import keras
import os

# parse optional arguments
parser = argparse.ArgumentParser(description='Find the best model parameter, by epoch, given a certain model.')
parser.add_argument('--model', type=str, default='cifar10')
parser.add_argument('--criterion', type=str, default='loss')
args = parser.parse_args()

model, criterion = args.model, args.criterion

# path to parameters during training
net_path = '../work/training/' + model + '/'


