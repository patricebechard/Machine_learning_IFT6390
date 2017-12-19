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
from scipy.io import loadmat

import matplotlib.pyplot as plt
from urllib.request import urlretrieve
import zipfile
import os
import sys

DATAZIP = 'data/training2017.zip'
DATADIR = 'data/training2017'
DATAURL = "https://physionet.org/challenge/2017/training2017.zip"
N_PTS = 2500
N_FEATURES = 300

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

    #translating letter labels to int labels (normal or abnormal only)
    intab = 'NAO~'
    outtab = '0111'
    transtab = str.maketrans(intab, outtab)

    # loading data from every .mat file
    for i in range(datafiles.shape[0]):
        path = DATADIR + '/' + datafiles[i,0] + '.mat'

        #not working anymore for an obscure reason
        features = loadmat(path)
        features = features['val']

        features = features[0,:N_PTS]           #keep only a little sample of pts
        features = features / np.std(features)  #dividing by std dev

        x_interp = np.linspace(0, N_PTS, N_FEATURES)
        # Reducing number of features
        features = np.interp(x_interp, np.arange(N_PTS), features)

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

    fmt = ['%.18e' for i in range(N_FEATURES)]
    fmt.append('%d')

    np.savetxt('data/ecg.train', features_table[:lim1], delimiter=',',fmt=fmt)
    np.savetxt('data/ecg.valid', features_table[lim1:lim2], delimiter=',',fmt=fmt)
    np.savetxt('data/ecg.test', features_table[lim2:], delimiter=',',fmt=fmt)

<<<<<<< HEAD
def show_ex():
    datafiles = np.genfromtxt(DATADIR + '/REFERENCE.csv',
                        dtype='str', delimiter=',')   
=======
def print_example():

    datafiles = np.genfromtxt(DATADIR + '/REFERENCE.csv',
                         dtype='str', delimiter=',')


>>>>>>> 01e5469d22c64a931ca1ca2b15d2c9f8159ae516

    #translating letter labels to int labels (normal or abnormal only)
    intab = 'NAO~'
    outtab = '0111'
    transtab = str.maketrans(intab, outtab)

<<<<<<< HEAD
=======
    for i in range(datafiles.shape[0]):
        datafiles[i,1] = datafiles[i,1].translate(transtab)
    labels = np.array([int(datafiles[i,-1]) for i in range(datafiles.shape[0])])
    print(np.bincount(labels))
    print(labels)
    exit()
   

>>>>>>> 01e5469d22c64a931ca1ca2b15d2c9f8159ae516
    # loading data from every .mat file
    for i in range(datafiles.shape[0]):
        path = DATADIR + '/' + datafiles[i,0] + '.mat'

        #not working anymore for an obscure reason
        features = loadmat(path)
        features = features['val']

<<<<<<< HEAD
        features = features[0,:N_PTS]           #keep only a little sample of pts
        features = features / np.std(features)  #dividing by std dev

        for npts in [100, 150, 200, 250, 300, 350, 400, 450, 500]:


            x_interp = np.linspace(0, N_PTS, npts)
            # Reducing number of features
            featuresnew = np.interp(x_interp, np.arange(N_PTS), features)

            plt.plot(featuresnew)
            plt.xlabel(str(npts))
            plt.show()
        exit()

        datafiles[i,1] = datafiles[i,1].translate(transtab)

=======
        print(features)
        plt.plot(features[0])
        plt.savefig('ecg_raw.png')
        plt.show()

        features = features[0,:N_PTS]           #keep only a little sample of pts
        features = features / np.std(features)  #dividing by std dev

        x_interp = np.linspace(0, N_PTS, N_FEATURES)
        # Reducing number of features
        features = np.interp(x_interp, np.arange(N_PTS), features)

        datafiles[i,1] = datafiles[i,1].translate(transtab)

        plt.plot(features)
        plt.savefig('ecg_norm.png')
        plt.show()
        exit()

>>>>>>> 01e5469d22c64a931ca1ca2b15d2c9f8159ae516
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
<<<<<<< HEAD
=======
    print
>>>>>>> 01e5469d22c64a931ca1ca2b15d2c9f8159ae516
    features_table = np.append(features_table, datafiles, axis=1)   #adding labels
    features_table = features_table.astype(np.float32)

    np.random.shuffle(features_table)                   #shuffle data

    # Splitting data in 3 distinct sets (training, valid, test)
    lim1 = 2 * features_table.shape[0] // 3
    lim2 = 5 * features_table.shape[0] // 6

    fmt = ['%.18e' for i in range(N_FEATURES)]
    fmt.append('%d')

<<<<<<< HEAD

if __name__ == "__main__":
    retrieve_data()
    preprocess_data()
    #show_ex()
=======
    np.savetxt('data/ecg.train', features_table[:lim1], delimiter=',',fmt=fmt)
    np.savetxt('data/ecg.valid', features_table[lim1:lim2], delimiter=',',fmt=fmt)
    np.savetxt('data/ecg.test', features_table[lim2:], delimiter=',',fmt=fmt)



if __name__ == "__main__":
    #retrieve_data()
    #preprocess_data()
    print_example()
>>>>>>> 01e5469d22c64a931ca1ca2b15d2c9f8159ae516
