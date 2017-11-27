#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Patrice Bechard
@email: bechardpatrice@gmail.com
Created on Mon Nov 27 13:09:23 2017

Deep neural network implementation using Keras
for sequential ecg data

"""

from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import sys

np.random.seed(69)
N_IN = 500
N_OUT = 2
n_epochs = 50

def convert_to_onehot(labels, n_classes):
    new_labels = np.zeros((len(labels), n_classes))
    for i in range(len(labels)):
        new_labels[i, labels[i]] = 1
    return new_labels

def train_model(training_set, valid_set, hidden_units,
                n_epochs=250, batch_size=10, n_prints=25):

    X_train = training_set[:,:-1]
    Y_train = convert_to_onehot(training_set[:,-1].astype('int'), N_OUT)

    X_valid = valid_set[:,:-1]
    Y_valid = convert_to_onehot(valid_set[:,-1].astype('int'), N_OUT)

    model = Sequential()

    # adding layers to neural network
    for i in range(len(hidden_units) + 1):
        if i == 0 :
            model.add(Dense(hidden_units[i],
                            input_dim=N_IN,
                            activation='relu'))
        elif i == len(hidden_units):
            model.add(Dense(N_OUT, activation='sigmoid'))
        else:
            model.add(Dense(hidden_units[i], activation='relu'))

    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    print("\nArchitecture of hidden layers : %s" % str(hidden_units))

    for i in range(n_prints):
        # Fit the model
        model.fit(X_train, Y_train, epochs=n_epochs//n_prints,
                  batch_size=batch_size, verbose=0)
        scores_train = model.evaluate(X_train, Y_train, verbose=0)
        scores_test = model.evaluate(X_valid, Y_valid, verbose=0)
        print("\nProgress : %d / %d epochs" %(i * n_epochs //n_prints, n_epochs))
        print("Train %s: %.2f%%" % (model.metrics_names[1],
                                      scores_train[1]*100))
        print("Valid %s: %.2f%%" % (model.metrics_names[1],
                                      scores_test[1]*100))



training_set = np.loadtxt("data/ecg.train", delimiter=',')
valid_set = np.loadtxt("data/ecg.valid", delimiter=',')
test_set = np.loadtxt("data/ecg.test", delimiter=',')

hidden_units_file = open("hidden_units.txt")

for line in hidden_units_file:

    hidden_units = line.strip().split(',')
    hidden_units = list(map(int, hidden_units))

    train_model(training_set, valid_set, hidden_units,
                n_epochs=n_epochs, n_prints=5)


