#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Patrice Bechard
@email: bechardpatrice@gmail.com
Created on Wed Nov  8 09:37:06 2017

DNN classifier for ECG data

"""

import numpy as np
import tensorflow as tf
import os

ECG_TRAIN = os.getcwd() + '/data/ecg.train'
ECG_VALID = os.getcwd() + '/data/ecg.valid'
ECG_TEST = os.getcwd() + '/data/ecg.test'

def ecg_input_fn(dataset, num_epochs=None, shuffle=True):
    return  tf.estimator.inputs.numpy_input_fn(x={"x": np.array(dataset.data)},
                                               y=np.array(dataset.target),
                                               num_epochs=num_epochs,
                                               shuffle=shuffle)

def classifier(n_steps, hidden_units):

    # load datasets
    training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
            filename=ECG_TRAIN,
            target_dtype=np.int32,
            features_dtype=np.float32)
    valid_set = tf.contrib.learn.datasets.base.load_csv_without_header(
            filename=ECG_VALID,
            target_dtype=np.int32,
            features_dtype=np.float32)

    feature_columns = [tf.feature_column.numeric_column("x", shape=[2500])]

    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                     hidden_units=hidden_units,
                                     n_classes=4)#,
                                     #model_dir="/tmp/ecg_model")

    train_input_fn = ecg_input_fn(training_set)

    # Train model
    classifier.train(input_fn=train_input_fn, steps=n_steps)

    valid_input_fn = ecg_input_fn(valid_set, num_epochs=1, shuffle=False)

    # Evaluate accuracy
    accuracy_score = classifier.evaluate(input_fn=valid_input_fn)["accuracy"]

    print(hidden_units)
    print("\n Accuracy : {0:f}\n".format(accuracy_score))

    return accuracy_score

n_steps = 2000

hidden_units_test = np.array([[10, 10],
                              [10, 20, 10],
                              [20, 20],
                              [20, 10, 20],
                              [50, 50],
                              [100, 50],
                              [100, 100]])

acc = []
for hidden_units in hidden_units_test:
    accuracy_score = classifier(n_steps, hidden_units)
    acc.append(accuracy_score)

print(acc)

