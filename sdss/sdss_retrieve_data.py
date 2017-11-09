#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Patrice Bechard
@email: bechardpatrice@gmail.com
Created on Wed Nov  8 16:32:33 2017

Retrieving and preprocessing data for star classifier
data from : https://dr14.sdss.org/

"""

import numpy as np
from urllib.request import urlretrieve
from astropy.io import fits
from scipy.signal import lfilter
import matplotlib.pyplot as plt
import os
import sys

"""
We first have to download every
"""

DATAURL = 'https://dr14.sdss.org/optical/spectrum/search/download/format%3Dcsv/id%3D43633'
STARDB = 'data/stardb.csv'
USEDSTARDB = 'data/usedstardb.csv'

SPECPATH = 'data/spec/'
#SPECPATH = '/Volumes/BACKUP/data/sdss/'
STARSURLDB = 'data/download_url.txt'

TYPES_COUNTER = {'A' : 0,'F' : 0, 'G' : 0,'K' : 0,'M' : 0,'WD' : 0}
KEPT_TYPES = {'A' : 0,'F' : 1, 'G' : 2,'K' : 3,'M' : 4,'WD' : 5}
N_STARS = 10000

def retrieve_data():

    print("Retrieving data")

    if not os.path.exists(STARDB):
        urlretrieve(DATAURL, STARDB)

    used_stardb = None
    star_url_db = None

    f = open(STARDB)
    num_stars = 0
    for line in f:
        num_stars += 1
    f.close()

    # associated label : A:0, F:1, G:2, K:3, M:4, WD:5
    f = open(STARDB)
    f.readline()    #skip header
    i = 0
    for line in f:
        i += 1

        star_info = line.strip().replace("'",'').split(',')
        sptype = star_info[-1]

        if sptype[-1] in "0123456789":      #we don't care about temperature
            sptype = sptype[:-1]

        if sptype not in TYPES_COUNTER:        #we don't care about those stars
            continue
        elif TYPES_COUNTER[sptype] >= N_STARS: #we already have enough of this type
            continue
        else:
            TYPES_COUNTER[sptype] += 1
        print("Progression : %d of %d stars in database" % (i, num_stars))

        plate = int(star_info[0])
        mjd = int(star_info[1])
        fiberid = int(star_info[2])
        fits_file = "spec-%04d-%05d-%04d.fits" % (plate, mjd, fiberid)
        fits_url = "https://data.sdss.org/sas/dr14/sdss/spectro/redux/26/" + \
                      "spectra/lite/%04d/%s"%(plate,fits_file)

        # download fits file
        #urlretrieve(fits_url, SPECPATH + fits_file)

        newstar = np.expand_dims(np.array([fits_file, sptype], dtype=np.str),
                                 axis=0)

        fits_url = np.expand_dims(np.array([fits_url], dtype=np.str), axis=0)

        if star_url_db is not None:
            star_url_db = np.append(star_url_db,
                                    np.array(fits_url, dtype=np.str),
                                    axis=0)
        else:
            star_url_db = fits_url

        if used_stardb is not None:
            used_stardb = np.append(used_stardb, newstar, axis=0)
        else:
            used_stardb = newstar

    np.savetxt(USEDSTARDB, used_stardb, fmt='%s', delimiter=',')
    np.savetxt(STARSURLDB, star_url_db, fmt='%s')

    f.close()

def preprocess_data():

    f = open(USEDSTARDB)
    f.readline()
    num_stars = N_STARS * len(KEPT_TYPES)
    i = 0

    star_features = None

    for line in f:
        i += 1
        print("Progression : %d of %d stars in database" % (i, num_stars))
        [fits_file, sptype] = line.strip().split(',')

        hdulist = fits.open(SPECPATH + fits_file)
        wv = hdulist['COADD'].data['loglam']
        spec = hdulist['COADD'].data['flux']

        #keep values with loglam between 3.60 and 3.85 (approx 3981 to 7079 angstrom)
        lower = np.where(wv >= 3.60)
        upper = np.where(wv <= 3.85)
        domain = np.intersect1d(lower, upper)
        spec = spec[domain]

        spec = smooth_spectrum(spec)

        spec = np.append(spec, KEPT_TYPES[sptype])

        if star_features is not None:
            spec = np.expand_dims(spec, axis=0)
            star_features = np.append(star_features, spec, axis=0)
        else:
            star_features = np.expand_dims(spec, axis=0)

    np.savetxt('data/prepared_data.txt',star_features)




def smooth_spectrum(data, n=15, a=1, npts=1000):

    b = [1.0 / n] * n

    yy = lfilter(b,a,data)                   #smoothens spectrum
    xx = np.arange(len(yy))

    x_interp = np.linspace(0, len(yy), npts)

    yy = np.interp(x_interp, xx, yy)
    xx = x_interp

    params = np.polyfit(xx, yy, 2)  #fit a 2nd degree polynomial
    nor = params[2] + xx * params[1] + (xx ** 2) * params[0]
    yy /= nor              #normalized spectrum (not so clean)

    yy -= np.mean(yy)               #center data to 0
    yy /= np.std(yy)                #scale between 1 and -1 mostly

    return yy



if __name__ == "__main__":
    #retrieve_data()
    preprocess_data()

