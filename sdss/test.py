#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Patrice Bechard
@email: bechardpatrice@gmail.com
Created on Wed Nov  8 12:08:36 2017

Insert title of code and description here

Info about .fits files
https://data.sdss.org/datamodel/files/BOSS_SPECTRO_REDUX/RUN2D/spectra/PLATE4/spec.html

range from 3.60 to 3.85 loglam (approx 3981 to 7079 angstrom)

"""

from astropy.io import fits

path = '/Volumes/BACKUP/data/sdss/'

hdulist = fits.open(path + 'spec-0266-51602-0034.fits')

#print(hdulist.info())

wv = hdulist['COADD'].data['loglam']
spec2 = hdulist['COADD'].data['flux']
