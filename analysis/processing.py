"""
Author: Nick Yu
Date created: 13/11/2018

Module for processing audio
"""

import numpy as np
import math
import scipy.io.wavfile
from scipy.fftpack import dct


def convert_to_mfcc(audio_file, frames=None, frame_size=0.025, frame_overlap=0.01, n_coeff=26, n_ceps=12):
    """
    Converts given audio file to mfcc
    :param audio_file: location of audio file
    :param frames: amount of frames to keep, if None keep all
    :param frame_size: the amount of ms a frame should be
    :param frame_overlap: the amount of ms a frame should overlap
    :param n_coeff: number of fourier coefficients to use
    :param n_ceps: number of mel ceps to return
    :return: matrix of mfccs
    """
    sample_rate, signal = scipy.io.wavfile.read(audio_file)

    frame_size = int(round(frame_size * sample_rate))
    frame_step = int(round(frame_overlap * sample_rate))

    # convert to mono audio
    signal = (signal[:, 0] + signal[:, 1]) / 2.0

    signal = convert_to_frame(signal, frame_size, frame_step)

    if frames and frames <= signal.shape[1]:
        signal = signal[:, :frames]

    # convert to complex numbers
    nfft = signal.shape[0]
    nfft_unique = math.floor(nfft / 2 + 1)
    result_signal = np.empty((nfft_unique, signal.shape[1]), dtype=complex)

    for i in range(signal.shape[1]):
        frame = signal[:, i] * np.hamming(frame_size)
        result_signal[:, i] = np.abs(np.fft.fft(frame)[:nfft_unique])

    # compute MFCCs
    filterbanks = create_filter_bank(sample_rate, n_coeff, nfft)

    filterbanks = np.dot(result_signal.T, filterbanks.T).T

    return np.nan_to_num(dct(np.log10(filterbanks), axis=0)[1:n_ceps + 1])


def convert_to_frame(signal, frame_length, frame_step):
    """Converts a signal into overlapping frames.
    If the signal is not a multitude of the frame step, it is zero-padded

    :param signal: Signal to split up
    :param frame_length: The amount of elements in each frame
    :param frame_step: The amount of elements to start from
    :return: two-dimensional numpy.ndarray with frames in the first axis
    """
    signal_len = len(signal)
    frame_count = math.ceil((signal_len - frame_length) / frame_step) + 1

    total_length = (frame_count - 1)*frame_step + frame_length
    padding = np.zeros((total_length - signal_len,))
    signal = np.concatenate((signal, padding))

    remaining_frames = [signal[i*frame_step:i*frame_step + frame_length] for i in range(1, frame_count)]

    # concatenate the first frame with the frames at steps
    signal = np.concatenate((
        [signal[:frame_length]],
        remaining_frames
    )).T

    return signal


def create_filter_bank(sample_rate, nbanks, nfft, lower_limit=0):
    """Create n Mel filterbanks.

    :param sample_rate: The sampling frequency of the signal
    :param nbanks: The amount of filterbanks to generate
    :param nfft: The number of fft coefficients used
    :param lower_limit: lower frequency limit, by default 0 Hz
    :return: numpy.ndarray of nfft sized filters
    """
    upper_limit = sample_rate / 2

    mel_freq = np.linspace(convert_mel(lower_limit), convert_mel(upper_limit), num=nbanks+2)

    # convert back to hertz and then map to fft bins
    # TODO: check why nfft+1 instead of nfft
    fft_bins = [math.floor((nfft + 1) * inverse_mel(m) / sample_rate) for m in mel_freq]

    filter_banks = np.zeros((nbanks, math.floor(nfft / 2 + 1)))

    for i in range(1, nbanks + 1):
        # lower, center and upper fft bin values
        f_m_min, f_m, f_m_plus = fft_bins[i-1:i+2]

        # increasing half
        for k in range(f_m_min, f_m):
            filter_banks[i - 1, k] = (k - f_m_min) / (f_m - f_m_min)

        # decreasing half
        for k in range(f_m, f_m_plus):
            filter_banks[i - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    return filter_banks


def pre_emphasis_filter(signal, pre_emphasis=0.97):
    """Returns the signal with a pre emphasis filter applied.

    :param signal: The signal to apply the filter to
    :param pre_emphasis: The pre_emphasis alpha value
    :return: resulting signal
    """
    return np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])


def convert_mel(f):
    """Converts the given frequency to Mel scale"""
    return 1125 * math.log(1 + f / 700)


def inverse_mel(m):
    """Converts the given Mels back to frequency"""
    return 700 * (math.exp(m / 1125) - 1)
