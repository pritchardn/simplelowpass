import matplotlib.pyplot as plt
import numpy as np
import pyfftw


def sinc(x: np.float):
    if np.isclose(x, 0.0):
        return 1.0
    else:
        return np.sin(np.pi * x) / (np.pi * x)


def gen_sig(freqs, length, sample_rate):
    x = np.zeros(length, dtype=np.float)
    for f in freqs:
        for i in range(length):
            x[i] += np.sin(2 * np.pi * i * f / sample_rate)
    return x


def window(length, cutoff, sample_rate):
    alpha = 2 * cutoff / sample_rate
    win = np.zeros(length)
    for i in range(length):
        ham = 0.54 - 0.46 * np.cos(2 * np.pi * i / length)  # Hamming coefficient
        hsupp = (i - length / 2)
        win[i] = ham * alpha * sinc(alpha * hsupp)
    return win


def determine_size(length):
    return int(2 ** np.ceil(np.log2(length)))


def fftea_time_np(signal: np.array, window: np.array):
    nfft = determine_size(len(signal) + len(window) - 1)
    xzp = np.zeros(nfft)
    hzp = np.zeros(nfft)
    xzp[:len(signal)] = signal
    hzp[:len(window)] = window
    X = np.fft.fft(xzp)
    H = np.fft.fft(hzp)
    Y = np.multiply(X, H)
    y = np.fft.ifft(Y)
    return np.real(y), np.imag(y)


if __name__ == "__main__":
    frequencies = [440, 880, 1000, 2000]
    M = 256  # Signal size
    L = 256  # Filter size
    cutoff_freq = 600
    sampling_rate = 5000
    sig = gen_sig(frequencies, M, sampling_rate)
    h = window(L, cutoff_freq, sampling_rate)
    np_filtered, error = fftea_time_np(sig, h)
    plt.plot(sig)
    plt.show()
    plt.plot(h)
    plt.show()
    plt.plot(np_filtered)
    plt.show()
