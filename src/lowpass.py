"""
Defines all low-pass filtering functions.
:param argv[1]: Whether to load the file directly (1) or not (else)
:param argv[2]: Input file base path
:param argv[3]: Output file base path
:param argv[4]: Precision double (1), single (else)
"""
import json
import sys
import pyfftw

import numpy as np

PRECISIONS = {'double': {'float': np.float64, 'complex': np.complex128},
              'single': {'float': np.float32, 'complex': np.complex64}}


def sinc(x_val: np.float64):
    """
    Computes the sin_c value for the input float
    :param x_val:
    """
    if np.isclose(x_val, 0.0):
        return 1.0
    return np.sin(np.pi * x_val) / (np.pi * x_val)


def gen_sig(freqs, length, sample_rate):
    """
    Generates a signal series composed of sine-waves.
    :param freqs: The list of frequency values (Hz)
    :param length: The length of the desired series
    :param sample_rate: The sample rate for data points
    :return: A numpy array signal series
    """
    series = np.zeros(length, dtype=np.float64)
    for freq in freqs:
        for i in range(length):
            series[i] += np.sin(2 * np.pi * i * freq / sample_rate)
    return series


def gen_window(length, cutoff, sample_rate):
    """
    Generates a filter-window sequence
    :param length: The length of the desired window
    :param cutoff: The cutoff frequency (Hz)
    :param sample_rate: The sampling rate
    :return: A numpy array window series
    """
    alpha = 2 * cutoff / sample_rate
    win = np.zeros(length)
    for i in range(length):
        ham = 0.54 - 0.46 * np.cos(2 * np.pi * i / length)  # Hamming coefficient
        hsupp = (i - length / 2)
        win[i] = ham * alpha * sinc(alpha * hsupp)
    return win


def add_noise(signal: np.array, mean, std, freq, sample_rate, seed, alpha=0.1):
    """
    A noise to the provided signal by producing random values of a given frequency
    :param signal: The input (and output) numpy array signal series
    :param mean: The average value
    :param std: The standard deviation of the value
    :param freq: The frequency of the noisy signal
    :param sample_rate: The sample rate of the input series
    :param seed: The random seed
    :param alpha: The multiplier
    :return: The input series with noisy values added
    """
    np.random.seed(seed)
    samples = alpha * np.random.normal(mean, std, size=len(signal))
    for i in range(len(signal)):
        samples[i] += np.sin(2 * np.pi * i * freq / sample_rate)
    np.add(signal, samples, out=signal)
    return signal


def determine_size(length):
    """
    :param length:
    :return: Computes the next largest power of two needed to contain |length| elements
    """
    return int(2 ** np.ceil(np.log2(length))) - 1


def filter_fft_np(signal: np.array, window: np.array, prec: dict):
    """
    Computes the low_pass filter using the numpy fft method
    :param signal: The input series
    :param window: The input window
    :param prec: The precision entry
    :return: The filtered signal
    """
    nfft = determine_size(len(signal) + len(window) - 1)
    sig_zero_pad = np.zeros(nfft, dtype=prec['float'])
    win_zero_pad = np.zeros(nfft, dtype=prec['float'])
    sig_zero_pad[0:len(signal)] = signal
    win_zero_pad[0:len(window)] = window
    sig_fft = np.fft.fft(sig_zero_pad)
    win_fft = np.fft.fft(win_zero_pad)
    out_fft = np.multiply(sig_fft, win_fft)
    out = np.fft.ifft(out_fft)
    return out.astype(prec['complex'])


def filter_fft_fftw(signal: np.array, window: np.array, prec: dict):
    """
    Computes the low_pass filter using the fftw fft method
    :param signal: The input series
    :param window: The input window
    :param prec: The precision entry
    :return: The filtered signal
    """
    nfft = determine_size(len(signal) + len(window) - 1)
    sig_zero_pad = pyfftw.empty_aligned(len(signal), dtype=prec['float'])
    win_zero_pad = pyfftw.empty_aligned(len(window), dtype=prec['float'])
    sig_zero_pad[0:len(signal)] = signal
    win_zero_pad[0:len(window)] = window
    sig_fft = pyfftw.interfaces.numpy_fft.fft(sig_zero_pad, n=nfft)
    win_fft = pyfftw.interfaces.numpy_fft.fft(win_zero_pad, n=nfft)
    out_fft = np.multiply(sig_fft, win_fft)
    out = pyfftw.interfaces.numpy_fft.ifft(out_fft, n=nfft)
    return out.astype(prec['complex'])


def filter_fft_cuda(signal: np.array, window: np.array, prec: dict):
    """
    Computes the low_pass filter using the numpy pycuda method.
    Also auto-inits the pycuda library
    :param signal: The input series
    :param window: The input window
    :param prec: The precision entry
    :return: The filtered signal
    """
    import pycuda.autoinit  # Here because it initialises a new cuda environment every trial.
    import pycuda.gpuarray as gpuarray
    import skcuda.fft as cu_fft
    import skcuda.linalg as linalg
    linalg.init()
    nfft = determine_size(len(signal) + len(window) - 1)
    # Move data to GPU
    sig_zero_pad = np.zeros(nfft, dtype=prec['float'])
    win_zero_pad = np.zeros(nfft, dtype=prec['float'])
    sig_gpu = gpuarray.zeros(sig_zero_pad.shape, dtype=prec['float'])
    win_gpu = gpuarray.zeros(win_zero_pad.shape, dtype=prec['float'])
    sig_zero_pad[0:len(signal)] = signal
    win_zero_pad[0:len(window)] = window
    sig_gpu.set(sig_zero_pad)
    win_gpu.set(win_zero_pad)

    # Plan forwards
    sig_fft_gpu = gpuarray.zeros(nfft, dtype=prec['complex'])
    win_fft_gpu = gpuarray.zeros(nfft, dtype=prec['complex'])
    sig_plan_forward = cu_fft.Plan(sig_fft_gpu.shape, prec['float'], prec['complex'])
    win_plan_forward = cu_fft.Plan(win_fft_gpu.shape, prec['float'], prec['complex'])
    cu_fft.fft(sig_gpu, sig_fft_gpu, sig_plan_forward)
    cu_fft.fft(win_gpu, win_fft_gpu, win_plan_forward)

    # Convolve
    out_fft = linalg.multiply(sig_fft_gpu, win_fft_gpu, overwrite=True)
    linalg.scale(2.0, out_fft)

    # Plan inverse
    out_gpu = gpuarray.zeros_like(out_fft)
    plan_inverse = cu_fft.Plan(out_fft.shape, prec['complex'], prec['complex'])
    cu_fft.ifft(out_fft, out_gpu, plan_inverse, True)
    out_np = np.zeros(len(out_gpu), prec['complex'])
    out_gpu.get(out_np)
    return out_np


def filter_pointwise_np(signal: np.array, window: np.array, prec: dict):
    """
    Computes the low_pass filter using the numpy point-wise convolution method
    :param signal: The input series
    :param window: The input window
    :param prec: The precision entry
    :return: The filtered signal
    """
    return np.convolve(signal, window, mode='full').astype(prec['complex'])


def main(fname, dirout, direct, precision):
    """
    Computes the low-pass filter response using each method implemented.
    :param fname: The relative input file path
    :param dirout: The relative base output file path
    :param direct: A boolean whether to use raw data files (true) or config files (false)
    :param precision: The numeric precision
    :return: Saves the filtered signal to a numpy file.
    """
    methods = {'numpy_fft': filter_fft_np, 'fftw': filter_fft_fftw, 'cufft': filter_fft_cuda,
               'numpy_pointwise': filter_pointwise_np}
    if precision == 1:
        prec = PRECISIONS['double']
    else:
        prec = PRECISIONS['single']
    for m_name, func in methods.items():
        if direct:
            container = np.load(fname)
            sig = container['sig']
            win = container['win']
            name = str(container['name'])
            if 'noise' in container.keys():
                outname = dirout + 'noisy/' + name + '_' + str(m_name) + '.out'
            else:
                outname = dirout + 'clean/' + name + '_' + str(m_name) + '.out'
        else:
            with open(fname, 'r') as file:
                config = json.load(file)
            sig = gen_sig(config['frequencies'], config['sig_len'], config['sampling_rate'])
            win = gen_window(config['win_len'], config['cutoff_freq'], config['sampling_rate'])
            outname = dirout + 'clean/' + config['name'] + '_' + str(m_name) + '.out'
            if 'noise' in config.keys():
                sig = add_noise(sig,
                                config['noise']['mean'],
                                config['noise']['std'],
                                config['noise']['frequency'],
                                config['sampling_rate'],
                                config['noise']['seed'],
                                config['noise']['alpha'])
                outname = dirout + 'noisy/' + config['name'] + '_' + str(m_name) + '.out'

        result = func(sig, win, prec)
        print("Saving to " + outname)
        np.save(outname, result)


if __name__ == "__main__":
    FILE_NAME = sys.argv[2]
    DIRECTORY_OUT = sys.argv[3]
    LOAD_DIRECT = bool(int(sys.argv[1]))
    if len(sys.argv) <= 4:
        PRECISION = PRECISIONS['double']  # default beahviour
    elif int(sys.argv[4]) == 1:
        PRECISION = PRECISIONS['double']
    else:
        PRECISION = PRECISIONS['single']
    main(FILE_NAME, DIRECTORY_OUT, LOAD_DIRECT, PRECISION)
