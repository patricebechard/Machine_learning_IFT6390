#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Patrice Bechard
@email: bechardpatrice@gmail.com
Created on Wed Dec 20 19:57:33 2017

Datasets for sdss

"""

import numpy as np

#datasets location
training_loc = "data/sdss.train"
valid_loc = "data/sdss.valid"
test_loc = "data/sdss.test"

class training:
    def __init__(self):
        """ Class used to load training set"""
        training_set = np.loadtxt(training_loc, delimiter=',')
        self.features = training_set[:,:-1]
        self.labels = training_set[:,-1].astype('int')

class valid:
    def __init__(self):
        """ Class used to load valid set"""
        valid_set = np.loadtxt(valid_loc, delimiter=',')
        self.features = valid_set[:,:-1]
        self.labels = valid_set[:,-1].astype('int')

class test:
    def __init__(self):
        """ Class used to load test set"""
        test_set = np.loadtxt(test_loc, delimiter=',')
        self.features = test_set[:,:-1]
        self.labels = test_set[:,-1].astype('int')

if __name__ == "__main__":

    training_set = training()

    print(training_set.features)
