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
from sklearn.decomposition import PCA
from numpy.fft import fft

import matplotlib.pyplot as plt
from urllib.request import urlretrieve
import zipfile
import os
import sys

DATAZIP = 'data/training2017.zip'
DATADIR = 'data/training2017'
DATAURL = "https://physionet.org/challenge/2017/training2017.zip"
N_PTS = 2500
N_FEATURES = 500

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

def print_example():

    datafiles = np.genfromtxt(DATADIR + '/REFERENCE.csv',
                         dtype='str', delimiter=',')



    #translating letter labels to int labels (normal or abnormal only)
    intab = 'NAO~'
    outtab = '0111'
    transtab = str.maketrans(intab, outtab)

    for i in range(datafiles.shape[0]):
        datafiles[i,1] = datafiles[i,1].translate(transtab)
    labels = np.array([int(datafiles[i,-1]) for i in range(datafiles.shape[0])])
    print(np.bincount(labels))
    print(labels)
    exit()


    # loading data from every .mat file
    for i in range(datafiles.shape[0]):
        path = DATADIR + '/' + datafiles[i,0] + '.mat'

        #not working anymore for an obscure reason
        features = loadmat(path)
        features = features['val']

        print(features)
        plt.plot(features[0])
        plt.savefig('ecg_raw.png')
        plt.show()

        features = features[0,:N_PTS]           #keep only a little sample of pts
        features = features / np.std(features)  #dividing by std dev

        x_interp = np.linspace(0, N_PTS, N_FEATURES)
        # Reducing number of features
        features = np.interp(x_interp, np.arange(N_PTS), features)

        plt.plot(features)
        plt.savefig('ecg_norm.png')
        plt.show()
        exit()

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
    print
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

def preprocess_PCA():

    #normalize data

    #crop data

    #transform using PCA

    datafiles = np.genfromtxt(DATADIR + '/REFERENCE.csv',
                         dtype='str', delimiter=',')

    #translating letter labels to int labels (normal or abnormal only)
    intab = 'NAO~'
    outtab = '0111'
    transtab = str.maketrans(intab, outtab)

    first = True

    for i in range(datafiles.shape[0]):
        path = DATADIR + '/' + datafiles[i,0] + '.mat'

        #not working anymore for an obscure reason
        features = loadmat(path)
        features = features['val'][0].astype('float32')

        if len(features) < 8500:
            datafiles[i,1] = -1
            continue

        features = features[:8500]
        features = features / np.std(features)  #dividing by std dev

        datafiles[i,1] = datafiles[i,1].translate(transtab)


        # putting everything in the same array
        if first:
            first = False
            features_table = np.expand_dims(features, axis=0)
        else:
            features_table = np.append(features_table,
                                       np.expand_dims(features, axis=0),
                                       axis=0)

        if i % 100 == 0 and i != 0:
            print("Progress : %d of %d files processed"
                  % (i, datafiles.shape[0]))

    #adding labels
    datafiles = datafiles[:,-1].astype('float')
    datafiles = datafiles[datafiles>=0]
    datafiles = np.expand_dims(datafiles, axis=1)

    features_table = np.append(features_table, datafiles, axis=1)
    features_table = features_table.astype(np.float32)

    np.random.shuffle(features_table)                   #shuffle data

    n_components_list = [20, 50, 100,  200, 500]

    for n_components in n_components_list:

        print("Number of principal components : %d"%n_components)

        labels = features_table[:,-1:]

        pca = PCA(n_components=n_components)
        temp = pca.fit_transform(features_table[:,:-1]) #without labels

        temp = np.append(temp, labels, axis=1)


        # Splitting data in 3 distinct sets (training, valid, test)
        lim1 = 2 * temp.shape[0] // 3
        lim2 = 5 * temp.shape[0] // 6

        fmt = ['%.18e' for i in range(N_FEATURES)]
        fmt.append('%d')

        np.savetxt('data/ecg_pca_%d.train'%n_components, temp[:lim1],
                   delimiter=',',fmt=fmt)
        np.savetxt('data/ecg._pca_%dvalid'%n_components, temp[lim1:lim2],
                   delimiter=',',fmt=fmt)
        np.savetxt('data/ecg_pca_%d.test'%n_components, temp[lim2:],
                   delimiter=',',fmt=fmt)

def preprocess_fourier():

    datafiles = np.genfromtxt(DATADIR + '/REFERENCE.csv',
                         dtype='str', delimiter=',')

    #translating letter labels to int labels (normal or abnormal only)
    intab = 'NAO~'
    outtab = '0111'
    transtab = str.maketrans(intab, outtab)

    # loading data from every .mat file

    first = True

    for i in range(datafiles.shape[0]):
        path = DATADIR + '/' + datafiles[i,0] + '.mat'

        #not working anymore for an obscure reason
        features = loadmat(path)
        features = features['val'][0].astype('float32')

        if len(features) < 8500:
            datafiles[i,1] = -1
            continue

        features = features[:8500]
        features = features / np.std(features)  #dividing by std dev

        datafiles[i,1] = datafiles[i,1].translate(transtab)


        # putting everything in the same array
        if first:
            first = False
            features_table = np.expand_dims(features, axis=0)
        else:
            features_table = np.append(features_table,
                                       np.expand_dims(features, axis=0),
                                       axis=0)

        if i % 100 == 0 and i != 0:
            print("Progress : %d of %d files processed"
                  % (i, datafiles.shape[0]))

    #adding labels
    datafiles = np.expand_dims(datafiles[:,-1], axis=1)
    print(len(datafiles))
    datafiles = datafiles[datafiles>=0]
    print(len(datafiles))

    features_table = np.append(features_table, datafiles, axis=1)
    features_table = features_table.astype(np.float32)

    np.random.shuffle(features_table)                   #shuffle data

    n_components_list = [20, 50, 100,  200, 500, 1000, 2000, 5000]

    for n_components in n_components_list:

        print("Number of Fourier components : %d"%n_components)

        labels = features_table[:,-1:]

        temp = fft(features_table, axis=1, n=n_components)

        temp = np.append(temp, labels, axis=1)


        # Splitting data in 3 distinct sets (training, valid, test)
        lim1 = 2 * temp.shape[0] // 3
        lim2 = 5 * temp.shape[0] // 6

        fmt = ['%.18e' for i in range(N_FEATURES)]
        fmt.append('%d')

        np.savetxt('data/ecg_pca_%d.train'%n_components, temp[:lim1],
                   delimiter=',',fmt=fmt)
        np.savetxt('data/ecg._pca_%dvalid'%n_components, temp[lim1:lim2],
                   delimiter=',',fmt=fmt)
        np.savetxt('data/ecg_pca_%d.test'%n_components, temp[lim2:],
                   delimiter=',',fmt=fmt)

if __name__ == "__main__":
    #retrieve_data()
    #preprocess_data()
    #print_example()
    preprocess_PCA()
    preprocess_fourier()