"""
A few utility functions for various low-pass filter trials
"""
import numpy as np


def normalize_signal(series):
    """
    Normalises the input series
    :param series: The input series
    :return: Normalised input series
    """
    # amean = np.mean(series)
    astd = np.std(series)
    series /= astd
    return series


def correlate(series_a, series_b):
    """
    Computes the cross correlation between two series
    :param series_a:
    :param series_b:
    :return: The cross-correlation between the two input series
    """
    return np.absolute(np.correlate(series_a, series_b, mode='valid') / len(series_a))
