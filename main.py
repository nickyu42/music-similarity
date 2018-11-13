"""
Author: Nick Yu
Date created: 11/11/2018
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile

from processing import convert_to_frame

AUDIO_FILE = 'crossing field.wav'

# The amount of ms a frame should be
FRAME_SIZE = 0.025

# The frame overlap
FRAME_OVERLAP = 0.010

sample_rate, signal = scipy.io.wavfile.read(AUDIO_FILE)

frame_size = int(round(FRAME_SIZE * sample_rate))
frame_step = int(round(FRAME_OVERLAP * sample_rate))

# convert to mono audio
signal = (signal[:, 0] + signal[:, 1]) / 2.0

# take first 3s
signal = signal[:int(3*sample_rate)]

# apply pre-emphasis filter
# pre_emphasis = 0.97
# signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

# convert each to frequency domain and apply hamming window
# w = lambda n, size: 0.53836 - 0.46164*np.cos(2*np.pi*n / (size - 1))

signal = convert_to_frame(signal, frame_size, frame_step)

# convert to complex numbers
nfft = signal.shape[0] // 2 + 1
result_signal = np.empty((nfft, signal.shape[1]), dtype=complex)

for i in range(signal.shape[1]):
    frame = signal[:, i] * np.hamming(frame_size)
    result_signal[:, i] = np.fft.fft(frame)[:nfft]

plt.pcolormesh(abs(result_signal))
plt.show()



