#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Patrice Bechard
@email: bechardpatrice@gmail.com
Created on Wed Dec 20 09:57:56 2017

SVM implementation for ecg classification

"""

from sklearn import svm
from sklearn.decomposition import PCA
import numpy as np
from numpy.fft import fft
import sys
import datasets


np.random.seed(69)

n_components_list = [0, 20, 40, 60, 80, 100]
kernels = ['linear', 'rbf', 'poly', 'sigmoid']

def preprocessing_PCA(training, valid, test, n_components):
    print("Number of principal components : %d"%n_components)

    pca = PCA(n_components=n_components)
    training.features = pca.fit_transform(training.features)
    valid.features = pca.fit_transform(valid.features)
    test.features = pca.fit_transform(test.features)

    return training, valid, test


def preprocessing_fourier(training, valid, test, n_features = 500):

    print("Preprocessing data with Fourier transform. n_features = %d"
                      %(n_features))

    training.features = fft(training.features, axis=1, n=n_features)
    valid.features = fft(valid.features, axis=1, n=n_features)
    test.features = fft(test.features, n=n_features)

    return training, valid, test


def fit_svm(training, valid, test, kernel='rbf', C=1.0):

    print('\nKernel : %s'%kernel)
    print('Penalty parameter C : %.3f'%(C))

    classifier = svm.SVC(kernel=kernel, C=C)
    classifier.fit(training.features, training.labels)

    train_pred = classifier.predict(training.features)
    valid_pred = classifier.predict(valid.features)
    test_pred = classifier.predict(test.features)

    train_accuracy = np.sum(train_pred == training.labels) / len(training.labels)
    valid_accuracy = np.sum(valid_pred == valid.labels) / len(valid.labels)
    test_accuracy = np.sum(test_pred == test.labels) / len(test.labels)

    print("Training accuracy   : %.2f%%"%(train_accuracy * 100))
    print("Validation accuracy : %.2f%%"%(valid_accuracy * 100))
    print("Test accuracy       : %.2f%%"%(test_accuracy * 100))

#for n in [20, 40, 60, 80, 100, 150, 200, 250, 500]:

for c in [0.01, 0.05, 0.1, 0.5, 1., 5., 10., 50., 100.]:
    training = datasets.training()
    valid = datasets.valid()
    test = datasets.test()

    #training, valid, test = preprocessing_PCA(training, valid, test, n_components)
    #training, valid, test = preprocessing_fourier(training, valid, test,
    #                                              n_features=n)

    fit_svm(training, valid, test, C=c)

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

"""
effect of PCA preprocessing on rbf kernel
-----------------------------------------
VERY FEW EFFECTS ON FINAL RESULT

n_components : 0    (no pca preprocessing)
training accuracy   : 87.74%
validation accuracy : 63.05%
test accuracy       : 59.28%

n_components : 20
Training accuracy   : 98.80%
Validation accuracy : 61.29%
Test accuracy       : 59.28%

n_components : 40
Training accuracy   : 99.65%
Validation accuracy : 62.56%
Test accuracy       : 58.51%

n_components : 60
Training accuracy   : 99.77%
Validation accuracy : 62.70%
Test accuracy       : 57.81%

n_components : 80
Training accuracy   : 99.72%
Validation accuracy : 62.28%
Test accuracy       : 59.00%

n_components : 100
Training accuracy   : 99.54%
Validation accuracy : 62.56%
Test accuracy       : 58.65%
"""

"""
Fourier transform for preprocessing on rbf kernel
-------------------------------------------------
Preprocessing data with Fourier transform. n_features = 20
Training accuracy   : 83.98%
Validation accuracy : 58.34%
Test accuracy       : 56.33%

Preprocessing data with Fourier transform. n_features = 40
Training accuracy   : 98.66%
Validation accuracy : 62.14%
Test accuracy       : 59.28%

Preprocessing data with Fourier transform. n_features = 60
Training accuracy   : 99.93%
Validation accuracy : 63.05%
Test accuracy       : 58.72%

Preprocessing data with Fourier transform. n_features = 80
Training accuracy   : 99.98%
Validation accuracy : 63.05%
Test accuracy       : 58.65%

Preprocessing data with Fourier transform. n_features = 100
Training accuracy   : 100.00%
Validation accuracy : 63.27%
Test accuracy       : 58.79%

Preprocessing data with Fourier transform. n_features = 150
Training accuracy   : 100.00%
Validation accuracy : 63.19%
Test accuracy       : 58.58%

Preprocessing data with Fourier transform. n_features = 200
Training accuracy   : 100.00%
Validation accuracy : 63.19%
Test accuracy       : 58.58%

Preprocessing data with Fourier transform. n_features = 250
Training accuracy   : 100.00%
Validation accuracy : 63.19%
Test accuracy       : 58.58%

Preprocessing data with Fourier transform. n_features = 500
Training accuracy   : 100.00%
Validation accuracy : 63.19%
Test accuracy       : 58.58%

"""