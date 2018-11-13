import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile

AUDIO_FILE = 'crossing field.wav'
FRAME_SIZE = 512

sample_rate, signal = scipy.io.wavfile.read(AUDIO_FILE)

#%% convert to mono audio
signal = (signal[:, 0] + signal[:, 1]) / 2.0

# take first 3s
signal = signal[:int(3*sample_rate)]

# apply pre-emphasis filter
pre_emphasis = 0.97
signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

#%% split into frames
padding = FRAME_SIZE - (len(signal) % FRAME_SIZE)
signal = np.append(signal, np.zeros(padding))
signal = np.array(np.split(signal, FRAME_SIZE))

#%% convert each to frequency domain and apply hamming window
# w = lambda n, size: 0.53836 - 0.46164*np.cos(2*np.pi*n / (size - 1))

# convert to complex numbers
signal = signal.astype(complex)

for i in range(signal.shape[1]):
    signal[:, i] *= np.hamming(FRAME_SIZE)
    signal[:, i] = np.fft.fft(signal[:, i])
    # plt.plot(abs(data[:, i]))


plt.pcolormesh(abs(signal))
plt.show()


