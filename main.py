"""
Author: Nick Yu
Date created: 11/11/2018
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
from scipy.fftpack import dct

from processing import convert_to_frame, create_filter_bank

AUDIO_FILE = 'data/crossing field.wav'

# The amount of ms a frame should be
FRAME_SIZE = 0.025

# The frame overlap
FRAME_OVERLAP = 0.010

# The number of cepstral coefficients to calculate
N_COEF = 26
N_CEPS = 12

sample_rate, signal = scipy.io.wavfile.read(AUDIO_FILE)

frame_size = int(round(FRAME_SIZE * sample_rate))
frame_step = int(round(FRAME_OVERLAP * sample_rate))

# convert to mono audio
signal = (signal[:, 0] + signal[:, 1]) / 2.0

# take first 3s
signal = signal[:int(3*sample_rate)]

# convert each to frequency domain and apply hamming window
# w = lambda n, size: 0.53836 - 0.46164*np.cos(2*np.pi*n / (size - 1))

signal = convert_to_frame(signal, frame_size, frame_step)

# convert to complex numbers
nfft = signal.shape[0]
nfft_unique = math.floor(nfft / 2 + 1)
result_signal = np.empty((nfft_unique, signal.shape[1]), dtype=complex)

for i in range(signal.shape[1]):
    frame = signal[:, i] * np.hamming(frame_size)
    result_signal[:, i] = np.abs(np.fft.fft(frame)[:nfft_unique])

# compute MFCCs
filterbanks = create_filter_bank(sample_rate, N_COEF, nfft)

filterbanks = np.dot(result_signal.T, filterbanks.T).T

# TODO
mfcc = dct(np.log10(filterbanks), axis=0)[1:N_CEPS + 1]

plt.pcolormesh(mfcc)
plt.show()



