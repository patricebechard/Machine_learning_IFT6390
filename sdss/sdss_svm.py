#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Patrice Bechard
@email: bechardpatrice@gmail.com
Created on Wed Dec 20 09:57:56 2017

SVM implementation for ecg classification

"""

from sklearn import svm
import numpy as np
import sys

np.random.seed(69)

N_OUT = 6


def onehot(labels, n_classes):
    return np.eye(n_classes)[labels]

training_set = np.loadtxt("data/sdss.train", delimiter=',')
valid_set = np.loadtxt("data/sdss.valid", delimiter=',')
test_set = np.loadtxt("data/sdss.test", delimiter=',')

X_train = training_set[:,:-1]
Y_train = training_set[:,-1].astype('int')

X_valid = valid_set[:,:-1]
Y_valid = valid_set[:,-1].astype('int')

X_test = test_set[:,:-1]
Y_test = test_set[:,-1].astype('int')

kernels = ['linear', 'rbf', 'poly', 'sigmoid']
for kernel in kernels :
    print('\nKernel : %s'%kernel)

    classifier = svm.SVC(decision_function_shape='ovo', kernel=kernel)
    classifier.fit(X_train, Y_train)

    train_pred = classifier.predict(X_train)
    valid_pred = classifier.predict(X_valid)
    test_pred = classifier.predict(X_test)

    train_accuracy = 1 - np.sum(train_pred == Y_train) / len(Y_train)
    valid_accuracy = 1 - np.sum(valid_pred == Y_valid) / len(Y_valid)
    test_accuracy = 1 - np.sum(test_pred == Y_test) / len(Y_test)

    print("Training accuracy   : %.2f%%"%(train_accuracy * 100))
    print("Validation accuracy : %.2f%%"%(valid_accuracy * 100))
    print("Test accuracy       : %.2f%%"%(test_accuracy * 100))


"""
Results
-------
linear svm :
    training accuracy   : 65.79%
    validation accuracy : 56.51%
    test accuracy       : 55.06%

rbf svm :
    training accuracy   : 87.74%
    validation accuracy : 63.05%
    test accuracy       : 59.28%

polynomial svm :
    training accuracy   : 97.87%
    validation accuracy : 59.25%
    test accuracy       : 55.06%

sigmoid svm :
    training accuracy   : 52.15%
    validation accuracy : 51.58%
    test accuracy       : 52.60%

"""