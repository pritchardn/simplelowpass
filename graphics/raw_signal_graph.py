import sys

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from lowpass import pointwise_np


def plot_signal(sig):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(sig)
    ax.set_xlim(0, 256)
    ax.set(title='Input signal (440, 880, 1000, 2000) Hz',
           ylabel='Amplitude',
           xlabel='Time (samples)')
    plt.savefig('signal.png')
    plt.show()


def plot_window(win):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(win)
    ax.set_xlim(0, 256)
    ax.set(title='600 Hz Low-pass Filter Hann Window',
           ylabel='Amplitude',
           xlabel='Time (samples)')
    plt.savefig('window.png')
    plt.show()


def plot_filtered(filtered):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.real(filtered))
    ax.set_xlim(0, 512)
    ax.set_ylim(-2, 2)
    ax.set(title='Filtered signal',
           ylabel='Amplitude',
           xlabel='Time (samples)')
    plt.savefig('filtered.png')
    plt.show()


if __name__ == '__main__':
    fname = '../data/0.in.npz'
    container = np.load(fname)
    signal = container['sig']
    window = container['win']
    name = str(container['name'])
    filtered = pointwise_np(signal, window)
    sns.set_theme()
    sns.set_context("paper")
    sns.set_style("white")
    sns.color_palette("colorblind")

    plot_window(window)
    plot_signal(signal)
    plot_filtered(filtered)
