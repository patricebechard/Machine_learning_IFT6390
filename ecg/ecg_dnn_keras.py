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

def convert_to_onehot(labels, n_classes):
    new_labels = np.zeros((len(labels), n_classes))
    for i in range(len(labels)):
        new_labels[i, labels[i]] = 1
    return new_labels

np.random.seed(69)

training_set = np.loadtxt("data/ecg.train", delimiter=',')

X = training_set[:,:-1]
Y = convert_to_onehot(training_set[:,-1].astype('int'), 4)



model = Sequential()
model.add(Dense(12, input_dim=2500, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


for i in range(10):
    # Fit the model
    model.fit(X, Y, epochs=10, batch_size=10, verbose=0)
    scores = model.evaluate(X, Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
