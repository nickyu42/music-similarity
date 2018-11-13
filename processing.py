"""
Author: Nick Yu
Date created: 13/11/2018
"""

import numpy as np
import math


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


def create_filter_bank(sample_rate, nbanks, nfft):
    """Create n Mel filterbanks.

    :param sample_rate: The sampling frequency of the signal
    :param nbanks: The amount of filterbanks to generate
    :param nfft: The number of fft coefficients used
    :return: numpy.ndarray of nfft sized filters
    """
    pass


def pre_emphasis_filter(signal, pre_emphasis=0.97):
    """Returns the signal with a pre emphasis filter applied.

    :param signal: The signal to apply the filter to
    :param pre_emphasis: The pre_emphasis alpha value
    :return: resulting signal
    """
    return np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
