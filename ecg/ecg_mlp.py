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
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import sys

np.random.seed(69)

N_IN = 500
N_OUT = 2
n_epochs = 30

def convert_to_onehot(labels, n_classes):
    return np.eye(n_classes)[labels]

def train_model(training_set, valid_set, hidden_units,
                n_epochs=250, batch_size=20, n_prints=25,
                actv='relu', optimizer='adam', reg=0.0001,
                regtype = None):

    if regtype == 'l1':
        regularizer = regularizers.l1(reg)
    elif regtype == 'l2':
        regularizer=regularizers.l2(reg)
    elif regtype == 'l1l2':
        regularizer=regularizers.l1_l2(reg)
    else:
        regularizer=None

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
                            activation=actv,
                            kernel_regularizer=regularizer))
        elif i == len(hidden_units):
            model.add(Dense(N_OUT, activation='softmax'))
        else:
            model.add(Dense(hidden_units[i], activation=actv,
                            kernel_regularizer=regularizer))

    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

    print("\nArchitecture of hidden layers : %s" % str(hidden_units))

    train_error = []
    test_error = []

    for i in range(n_prints):

        scores_train = model.evaluate(X_train, Y_train, verbose=0)
        scores_test = model.evaluate(X_valid, Y_valid, verbose=0)

        print("\nProgress : %d / %d epochs" %(i * n_epochs //n_prints, n_epochs))
        print("Train %s: %.2f%%" % (model.metrics_names[1],
                                      scores_train[1]*100))
        print("Valid %s: %.2f%%" % (model.metrics_names[1],
                                      scores_test[1]*100))

        # Fit the model
        model.fit(X_train, Y_train, epochs=n_epochs//n_prints,
                  batch_size=batch_size, verbose=0)

        train_error.append(1 - scores_train[1])
        test_error.append(1 - scores_test[1])

        pred = model.predict(X_valid)
        pred = np.argmax(pred, axis=1)
        conf = np.zeros((2,2))

        for i in range(len(pred)):
            conf[pred[i],np.argmax(Y_valid[i])] += 1

        print(conf)

        # Fit the model
        model.fit(X_train, Y_train, epochs=1,
                  batch_size=batch_size, verbose=0)

    print("\nResults")
    print("Train %s: %.2f%%" % (model.metrics_names[1],
                                  scores_train[1]*100))
    print("Valid %s: %.2f%%" % (model.metrics_names[1],
                                  scores_test[1]*100))

    train_error.append(1 - scores_train[1])
    test_error.append(1 - scores_test[1])

    train_error = np.array(train_error)
    test_error = np.array(test_error)

    np.savetxt('training.txt',train_error)
    np.savetxt('test.txt',test_error)


    archstr = str(hidden_units).strip('[]').replace(' ','_').replace(',','')

    case_str = "ecg_%s_%s_%f"%(archstr, regtype, reg)
    case_str_train = "ecg_train_%s_%s_%f"%(archstr, regtype, reg)
    case_str_test = "ecg_test_%s_%s_%f"%(archstr, regtype, reg)

    plt.plot(np.arange(n_epochs+1), train_error, 'k', label='Train error')
    plt.plot(np.arange(n_epochs+1), test_error, 'k--', label='Test error')
    plt.xlabel("Number of training epochs")
    plt.ylabel("Classification error")
    plt.legend(fancybox=True, shadow=True)
    plt.title("Architecture of hidden layers : %s" % str(hidden_units))
    #plt.savefig("figures/ecg_%s.png"%archstr)
    plt.savefig("figures/ecg_%s.png"%(case_str))
    plt.clf()

    np.savetxt("log/%s.txt"%(case_str_train), train_error)
    np.savetxt("log/ecg_%s.txt"%(case_str_test), test_error)

# ----------------------------MAIN-----------------------------------------

#training_set = np.loadtxt("data/ecg.train", delimiter=',')
#valid_set = np.loadtxt("data/ecg.valid", delimiter=',')
#test_set = np.loadtxt("data/ecg.test", delimiter=',')

training_set = np.loadtxt("data/ecg_fourier_500.train", delimiter=',')
valid_set = np.loadtxt("data/ecg_fourier_500.valid", delimiter=',')
test_set = np.loadtxt("data/ecg_fourier_500.test", delimiter=',')

hidden_units_file = open("hidden_units.txt")

for line in hidden_units_file:

    hidden_units = line.strip().split(',')
    hidden_units = list(map(int, hidden_units))

    #regs = [0., 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    #regtypes = ['l1', 'l2', 'l1l2']

    regs = [0.0005]
    regtypes = ['l1l2']

    for reg in regs:
        print("Regularization value : %f:"%reg)
        for regtype in regtypes:
            train_model(training_set, valid_set, hidden_units,
                        n_epochs=n_epochs, n_prints=n_epochs, reg=reg,
                        regtype=regtype)