#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Patrice Bechard
@email: bechardpatrice@gmail.com
Created on Tue Nov  7 21:36:45 2017

Retrieving and preprocessing data for ECG
data from : https://physionet.org/challenge/2017/

"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
import zipfile
import os
import sys

DATAZIP = 'data/training2017.zip'
DATADIR = 'data/training2017'
DATAURL = "https://physionet.org/challenge/2017/training2017.zip"
N_PTS = 2500

def retrieve_data():

    # download data if not already on computer
    if not ((os.path.exists(DATAZIP) or os.path.exists(DATADIR))):
        urlretrieve(DATAURL, DATAZIP)

    # unzip if not yet unzipped (delete zip file afterwards)
    if not os.path.exists(DATADIR):
        zip_ref = zipfile.ZipFile(DATAZIP)
        zip_ref.extractall('data/')
        zip_ref.close()
        os.remove(DATAZIP)

def preprocess_data():

    datafiles = np.genfromtxt(DATADIR + '/REFERENCE.csv',
                         dtype='str', delimiter=',')

    #translating letter labels to int labels
    intab = 'NAO~'
    outtab = '0123'
    transtab = str.maketrans(intab, outtab)

    # loading data from every .mat file
    for i in range(datafiles.shape[0]):
        path = DATADIR + '/' + datafiles[i,0] + '.mat'
        features = sp.io.loadmat(path)
        features = features['val']
        features = features[0,:N_PTS]           #keep only a little sample of pts
        features = features / np.std(features)  #dividing by std dev
        datafiles[i,1] = datafiles[i,1].translate(transtab)

        # putting everything in the same array
        if i == 0:
            features_table = np.expand_dims(features, axis=0)
        else:
            features_table = np.append(features_table,
                                       np.expand_dims(features, axis=0),
                                       axis=0)
        if i % 100 == 0 and i != 0:
            print("Progress : %d of %d files processed"
                  % (i, datafiles.shape[0]))

    datafiles = np.expand_dims(datafiles[:,-1], axis=1)
    features_table = np.append(features_table, datafiles, axis=1)   #adding labels
    features_table = features_table.astype(np.float32)

    np.random.shuffle(features_table)                   #shuffle data

    # Splitting data in 3 distinct sets (training, valid, test)
    lim1 = 2 * features_table.shape[0] // 3
    lim2 = 5 * features_table.shape[0] // 6

    np.savetxt('data/ecg.train', features_table[:lim1])
    np.savetxt('data/ecg.valid', features_table[lim1:lim2])
    np.savetxt('data/ecg.test', features_table[lim2:])

if __name__ == "__main__":
    retrieve_data()
    preprocess_data()
