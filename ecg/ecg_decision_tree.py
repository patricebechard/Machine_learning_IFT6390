#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Patrice Bechard
@email: bechardpatrice@gmail.com
Created on Mon Nov 27 21:52:08 2017

Decision tree classifier for ECG using Scikit-Learn

"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

training_set = np.loadtxt("data/ecg.train", delimiter=',')
valid_set = np.loadtxt("data/ecg.valid", delimiter=',')
test_set = np.loadtxt("data/ecg.test", delimiter=',')

classifier = DecisionTreeClassifier(random_state=0)

x_train = training_set[:,:-1]
y_train = training_set[:,-1]

score = cross_val_score(classifier, x_train, y_train, cv=10)

print(score)
