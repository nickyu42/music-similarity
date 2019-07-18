"""
Author: Nick Yu
Date created: 11/11/2018
"""

import numpy as np
import matplotlib.pyplot as plt

from processing import convert_to_mfcc
from classification import calculate_distance, gmm_js

mfcc1 = convert_to_mfcc('data/songs/Akagami no Shirayuki-hime.wav', frames=500)
mfcc2 = convert_to_mfcc('data/songs/Dragon Ball.wav', frames=500)

plt.pcolormesh(np.real(mfcc1))
plt.show()

dist = calculate_distance(mfcc1, mfcc2, gmm_js)


