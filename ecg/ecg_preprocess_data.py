#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Patrice Bechard
@email: bechardpatrice@gmail.com
Created on Tue Nov  7 21:36:45 2017

ecg

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def show_length_hist(lengths, n_bins=20):
    plt.hist(lengths, n_bins)
    plt.show()

def show_data(data):
    plt.plot(data)
    plt.show()

def preprocess_data():

    paths = [os.getcwd() + '/data/ecg/normal/',
             os.getcwd() + '/data/ecg/abnormal/']

    lengths = []
    init_datatable = True

    for path in paths:

        #defining target
        if path == paths[0]:
            target = 0
        else:
            target = 1

        for file in os.listdir(path):

            #load and keep only relevant data
            example = np.loadtxt(path + file)
            example = example[:,-1:]
            example /= np.std(example)
            n_pts = len(example)
            if n_pts >= 50:
                example = example[:50]  #crop all examples to 50 data pts
            else:
                continue                #else we don't keep the example

            #adding target at the end of the example
            example = np.append(example, [[target]], axis=0).T

            #info for histogram
            lengths.append(n_pts)

            #putting all in same array
            if init_datatable:
                datatable = example
                init_datatable = False
            else:
                datatable = np.append(datatable,example,axis=0)

        print(datatable.shape)
        #different table for each type of data classes
        if path == paths[0]:
            datatable_normal = datatable
            init_datatable = True
        else:
            datatable_abnormal = datatable

    show_length_hist(lengths)

    #splitting in training, validation and test sets
    n_ex_normal = datatable_normal.shape[0]
    n_ex_abnormal = datatable_abnormal.shape[0]

    training_set = np.append(datatable_normal[:(n_ex_normal//2)],
                             datatable_abnormal[:(n_ex_abnormal//2)],
                             axis=0)
    valid_set = np.append(datatable_normal[(n_ex_normal//2):(3*n_ex_normal//4)],
                          datatable_abnormal[(n_ex_abnormal//2):(3*n_ex_abnormal//4)],
                          axis=0)
    test_set = np.append(datatable_normal[(3*n_ex_normal//4):],
                         datatable_abnormal[(3*n_ex_abnormal//4):],
                         axis=0)

    np.random.shuffle(training_set)
    np.random.shuffle(valid_set)
    np.random.shuffle(test_set)

    savepath = os.getcwd() + '/data/ecg/'


    np.savetxt(savepath + 'ecg.train', training_set, delimiter=',')
    np.savetxt(savepath + 'ecg.valid', valid_set, delimiter=',')
    np.savetxt(savepath + 'ecg.test', test_set, delimiter=',')

preprocess_data()