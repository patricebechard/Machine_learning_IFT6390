#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import json
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Convolution1D, MaxPooling1D, Dense, Flatten
from keras.models import Sequential
from keras import regularizers

import sys

np.random.seed(69)

n_features = 1000
n_classes = 6
filter_length = 4
pool_size = 4
n_epochs = 30
batch_size=50

def to_onehot(labels, n_classes):
	"""
	"""

	return np.eye(n_classes)[labels.astype(int)]

class ConvNeuralNetwork1D:
	"""1D Convolutional neural network classifier using the Keras library.
	"""
	def __init__(self):
		self.built = False
		self.trained = False
		
	def build(self,
			input_dim, output_dim,
			n_filters, filter_length, cnn_activations, pool_size,
			dnn_layers_dim, dnn_activations, batch_size=50,
			loss='categorical_crossentropy', optimizer='adam'):
		"""Builds and compiles a 1D convolutional neural networks.
		
		# Arguments
			n_filters: Number of outputs of filters in the convolution. int.
			ker_size: Size of the 1D convolutional window. int.
			pool_size: Size of the max pooling window. int.
			n_outputs: Number of outputs computed by the network. Corresponds to
				the number of classes. int.
		"""
		self.model = Sequential()
		
		# Convolutional layers
		if len(cnn_activations) == 1:
			self.model.add(Convolution1D(n_filters[0], filter_length, activation=cnn_activations[0], input_shape=(input_dim, 1)))
			self.model.add(MaxPooling1D(pool_size=pool_size))
		else:
			self.model.add(Convolution1D(n_filters[0], filter_length, activation=cnn_activations[0], input_shape=(input_dim, 1)))
			self.model.add(MaxPooling1D(pool_size=pool_size))
			for i in range(len(cnn_activations)-1):
				self.model.add(Convolution1D(n_filters[i], filter_length, activation=cnn_activations[i]))
				self.model.add(MaxPooling1D(pool_size=pool_size))
		
		self.model.add(Flatten())
		
		# Dense neural network layers
		if len(dnn_activations) == 1:
			self.model.add(Dense(output_dim, activation=dnn_activations[0]))
		else:
			for i in range(len(dnn_activations)-1):
				self.model.add(Dense(dnn_layers_dim[i], activation=dnn_activations[i]))
			self.model.add(Dense(output_dim, activation=dnn_activations[-1]))
			
		self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

		self.built = True
	
	def train(self,
			x_train, y_train,
			x_test, y_test,
    		n_epochs,
    		show_acc=True):
		"""Trains the 1d convolutional neural network.
		
    	# Arguments
    		x_train: Training data. ndarray of 1D ndarray.
    		y_train: Training labels encoded in onehot. ndarray array of ndarray
    			of size equal to the number of classes.
    		x_test: Test data encoded in onehot. ndarray of 1D ndarray.
			y_test: Training labels encoded in onehot. ndarray array of ndarray
				of size equal to the number of classes.
			n_epochs: int.
			show_acc: Show accuracy plot. bool.
			
		# Raises
			RuntimeError: One must build the model before training it.
		"""
		if not self.built:
			raise RuntimeError('You haven\'t build your network yet.')
		
		logs = self.model.fit(x_train, y_train,
				epochs=n_epochs, batch_size=batch_size,
				validation_data=(x_test, y_test),
				verbose=2)
				
		self.trained = True
		
		if show_acc:
			plt.figure()
			plt.plot(np.arange(0, n_epochs), logs.history['acc'], markersize=0.2, label='training accuracy')
			plt.plot(np.arange(0, n_epochs), logs.history['val_acc'], markersize=0.2, label='test accuracy')
			plt.xlabel('epochs')
			plt.ylabel('accuracy')
			plt.xlim(0, n_epochs-1)
			plt.ylim(0.5, 1)
			plt.legend(loc='best')
			plt.savefig('acc_plot.pdf')
		
		pred = self.model.predict(x_test)
		
		print(1 - np.abs(np.argmax(pred, axis=1) - np.argmax(y_test, axis=1)).mean())
	
	def predict(self, x):
		"""Computes the model's prediction for some 1D array of data.
		
		# Arguments
			x: 1D ndarray.
			
		# Raises
			RuntimeError: One must build the model before training it.
		"""
		if not self.trained:
			raise RuntimeError('You haven\'t trained your network yet.')
			
		return np.argmax(self.model.predict(x), axis=1)

if __name__ == "__main__":
	# Loading data
	training_set = np.loadtxt("data/sdss.train", delimiter=',')
	valid_set = np.loadtxt("data/sdss.valid", delimiter=',')
	test_set = np.loadtxt("data/sdss.test", delimiter=',')
	
	x_train = np.expand_dims(training_set[:, :-1], axis=2)
	y_train = to_onehot(training_set[:, -1], n_classes)
	x_valid = np.expand_dims(valid_set[:, :-1], axis=2)
	y_valid = to_onehot(valid_set[:, -1], n_classes)
	x_test = np.expand_dims(test_set[:, :-1], axis=2)
	y_test = to_onehot(test_set[:, -1], n_classes)
	
	with open('architectures_cnn.json') as json_file:
		architectures = json.load(json_file)
	
	for architecture in architectures:

		print("ARCHITECTURE : %s"%str(architecture))
		cnn = ConvNeuralNetwork1D()
		cnn.build(n_features, n_classes, architecture['n_filters'],
	 		  filter_length, architecture['activations_cnn'],
	 		  pool_size, architecture['hid_layers_dim'], 
	 		  architecture['activations_dnn'], batch_size=batch_size)
		cnn.train(x_train, y_train, x_valid, y_valid, n_epochs)
