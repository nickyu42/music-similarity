"""
Author: Nick Yu
Date created: 11/11/2018
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

from processing import convert_to_mfcc
from classification import gmm_js, fit_gmm, init_gmm

samples = list()
samples.append(convert_to_mfcc('data/songs/Akagami no Shirayuki-hime.wav', frames=3000))
samples.append(convert_to_mfcc('data/songs/Dragon Ball.wav', frames=3000))

# plt.pcolormesh(np.real(mfcc1))
# plt.show()

gram_matrix = np.zeros((2, 2))

# precompute d_js
for i, x in enumerate(samples):
    for j, y in enumerate(samples):
        gram_matrix[i, j] = gmm_js(init_gmm(fit_gmm(x)), init_gmm(fit_gmm(y)))


# apply custom rbf kernel
def kernel(js: float, gamma: float = 0.1):
    return np.exp(-gamma * js)


gram_matrix = np.vectorize(kernel)(gram_matrix)

svc = svm.SVC(kernel='precomputed')
svc.fit(gram_matrix, [1, 2])

res = svc.predict(gram_matrix[1, :].reshape(1, -1))

