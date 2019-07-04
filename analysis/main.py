"""
Author: Nick Yu
Date created: 11/11/2018
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
from scipy.fftpack import dct

from processing import convert_to_mfcc
from classification import fit_gmm, init_gmm, gmm_js
from sklearn import mixture

mfcc1 = convert_to_mfcc('data/songs/Akagami no Shirayuki-hime.wav', frames=500)
mfcc2 = convert_to_mfcc('data/songs/Dragon Ball.wav')

plt.pcolormesh(np.real(mfcc1))
plt.show()

gmix1 = mixture.GaussianMixture(n_components=20, covariance_type='full')
gmix1.fit(np.nan_to_num(mfcc1.real).T)

gmix2 = mixture.GaussianMixture(n_components=20, covariance_type='full')
gmix2.fit(np.nan_to_num(mfcc2.real).T)

print(gmm_js(gmix1, gmix2))


