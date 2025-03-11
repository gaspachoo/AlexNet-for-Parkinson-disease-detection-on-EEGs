import numpy as np
from scipy.signal import butter, filtfilt

def butter_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=500.0, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=0)
    return y