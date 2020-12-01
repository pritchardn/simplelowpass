"""
Generates plots of the signal, window and filter series in the provided data file.
The data-file path is specified as a command line argument and should be a relative path.
The published example file (in UNIX file paths is) ../data/0.in.npz
"""
import sys

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rcParams

from lowpass import filter_pointwise_np


def plot_signal(signal):
    """
    Plots a signal series
    :param signal: The signal series
    """
    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.plot(signal)
    axis.set_xlim(0, len(signal))
    axis.set_ylim(-3.5, 3.5)
    axis.set(title='Input signal (440, 880, 1000, 2000) Hz',
             ylabel='Amplitude',
             xlabel='Time (samples)')
    plt.savefig('signal.png', dpi=300)
    plt.show()


def plot_window(window):
    """
    Plots a window series
    :param window: The signal series
    """
    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.plot(window)
    axis.set_xlim(0, len(window))
    axis.set(title='600 Hz Low-pass Filter Hann Window',
             ylabel='Amplitude',
             xlabel='Time (samples)')
    plt.savefig('window.png', dpi=300)
    plt.show()


def plot_filtered(filtered_signal):
    """
    Plots a filtered signal series
    :param filtered_signal: The signal series
    """
    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.plot(np.real(filtered_signal))
    axis.set_xlim(0, len(filtered_signal))
    axis.set_ylim(-2, 2)
    axis.set(title='Filtered signal',
             ylabel='Amplitude',
             xlabel='Time (samples)')
    plt.savefig('filtered.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    rcParams.update({'figure.autolayout': True})
    FNAME = sys.argv[1]
    CONTAINER = np.load(FNAME)
    SIGNAL = CONTAINER['sig']
    WINDOW = CONTAINER['win']
    NAME = str(CONTAINER['name'])
    FILTERED = filter_pointwise_np(SIGNAL, WINDOW, {'float': np.float64, 'complex': np.complex128})
    sns.set_theme()
    sns.set_context("paper")
    sns.set(font_scale=1.4)
    sns.set_style("white")
    sns.color_palette("colorblind")

    plot_window(WINDOW)
    plot_signal(SIGNAL)
    plot_filtered(FILTERED)
