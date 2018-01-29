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
import sys
import datasets

np.random.seed(69)
n_classes = 6

def preprocessing_PCA(training, valid, test, n_components):
    print("\nNumber of principal components : %d"%n_components)

    if n_components == 0:
        #no pca preprocessing
        return training, valid, test

    pca = PCA(n_components=n_components)
    training.features = pca.fit_transform(training.features)
    valid.features = pca.fit_transform(valid.features)
    test.features = pca.fit_transform(test.features)

    return training, valid, test

def fit_svm(training, valid, test, kernel='rbf', C=1.0, decision='ovr'):

    print('Kernel : %s'%kernel)
    print('Penalty parameter C : %.3f'%(C))

    classifier = svm.SVC(kernel=kernel, C=C, verbose=True,
                         decision_function_shape=decision)
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

    print(np.bincount(test_pred))
    print(np.bincount(test.labels))

#for n in [0, 20, 40, 60, 80, 100, 150, 200, 250, 500]:

training = datasets.training()
valid = datasets.valid()
test = datasets.test()

#training, valid, test = preprocessing_PCA(training, valid, test)
                                           #   n_components=n)
#training, valid, test = preprocessing_fourier(training, valid, test,
#                                              n_features=n)
for c in [10., 50., 100.]:
    print("C: %f"%c)
    fit_svm(training, valid, test, decision='ovr', C=c)



"""
Results 
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

Effect of number of principal components with PCA preprocessing (rbf kernel)

number of principal components : 0 (No PCA preprocessing)
    training accuracy   : 93.26%
    validation accuracy : 92.41%
    test accuracy       : 91.89%

number of principal components : 20
    training accuracy   : 97.81%
    validation accuracy : 64.49%
    test accuracy       : 58.08%

number of principal components : 40
    training accuracy   : 97.86%
    validation accuracy : 74.19%
    test accuracy       : 70.11%

number of principal components : 60
    training accuracy   : 97.75%
    validation accuracy : 77.43%
    test accuracy       : 73.91%

number of principal components : 80 
    training accuracy   : 97.60%
    validation accuracy : 79.31%
    test accuracy       : 76.17%

number of principal components : 100
    training accuracy   : 97.38%
    validation accuracy : 81.05%
    test accuracy       : 77.57%

number of principal components : 150
    training accuracy   : 96.95%
    validation accuracy : 83.29%
    test accuracy       : 80.55%

number of principal components : 200
    training accuracy   : 96.59%
    validation accuracy : 84.65%
    test accuracy       : 82.29%

number of principal components : 250
    training accuracy   : 96.23%
    validation accuracy : 84.97%
    test accuracy       : 82.76%

number of principal components : 500
    training accuracy   : 94.90%
    validation accuracy : 84.76%
    test accuracy       : 82.45%

"""
