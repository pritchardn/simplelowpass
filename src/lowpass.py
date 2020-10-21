import json
import sys

import numpy as np
from matplotlib import pyplot as plt


def sinc(x: np.float64):
    if np.isclose(x, 0.0):
        return 1.0
    else:
        return np.sin(np.pi * x) / (np.pi * x)


def gen_sig(freqs, length, sample_rate):
    x = np.zeros(length, dtype=np.float64)
    for f in freqs:
        for i in range(length):
            x[i] += np.sin(2 * np.pi * i * f / sample_rate)
    return x


def gen_window(length, cutoff, sample_rate):
    alpha = 2 * cutoff / sample_rate
    win = np.zeros(length)
    for i in range(length):
        ham = 0.54 - 0.46 * np.cos(2 * np.pi * i / length)  # Hamming coefficient
        hsupp = (i - length / 2)
        win[i] = ham * alpha * sinc(alpha * hsupp)
    return win


def add_noise(signal: np.array, mean, std, freq, sample_rate, seed, alpha=0.1):
    np.random.seed(seed)
    samples = alpha * np.random.normal(mean, std, size=len(signal))
    for i in range(len(signal)):
        samples[i] += np.sin(2 * np.pi * i * freq / sample_rate)
    np.add(signal, samples, out=signal)
    return signal


def determine_size(length):
    return int(2 ** np.ceil(np.log2(length))) - 1


def fftea_time_np(signal: np.array, window: np.array):
    nfft = determine_size(len(signal) + len(window) - 1)
    sig_zero_pad = np.zeros(nfft)
    win_zero_pad = np.zeros(nfft)
    sig_zero_pad[0:len(signal)] = signal
    win_zero_pad[0:len(window)] = window
    sig_fft = np.fft.fft(sig_zero_pad)
    win_fft = np.fft.fft(win_zero_pad)
    out_fft = np.multiply(sig_fft, win_fft)
    out = np.fft.ifft(out_fft)
    return out


def fftea_time_fftw(signal: np.array, window: np.array):
    import pyfftw
    nfft = determine_size(len(signal) + len(window) - 1)
    sig_zero_pad = pyfftw.empty_aligned(len(signal), dtype='float64')
    win_zero_pad = pyfftw.empty_aligned(len(window), dtype='float64')
    sig_zero_pad[0:len(signal)] = signal
    win_zero_pad[0:len(window)] = window
    sig_fft = pyfftw.interfaces.numpy_fft.fft(sig_zero_pad, n=nfft)
    win_fft = pyfftw.interfaces.numpy_fft.fft(win_zero_pad, n=nfft)
    out_fft = np.multiply(sig_fft, win_fft)
    out = pyfftw.interfaces.numpy_fft.ifft(out_fft, n=nfft)
    return out


def fftea_time_cuda(signal: np.array, window: np.array):
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    import skcuda.fft as cu_fft
    import skcuda.linalg as linalg
    ctx = pycuda.autoinit.make_default_context()
    ctx.pop()
    linalg.init()
    nfft = determine_size(len(signal) + len(window) - 1)
    # Move data to GPU
    sig_zero_pad = np.zeros(nfft)
    win_zero_pad = np.zeros(nfft)
    sig_zero_pad[0:len(signal)] = signal
    win_zero_pad[0:len(window)] = window
    sig_gpu = gpuarray.empty(sig_zero_pad.shape, dtype=np.float64)
    win_gpu = gpuarray.empty(win_zero_pad.shape, dtype=np.float64)
    sig_gpu.set(sig_zero_pad)
    win_gpu.set(win_zero_pad)

    # Plan forwards
    sig_fft_gpu = gpuarray.empty(nfft, np.complex128)
    win_fft_gpu = gpuarray.empty(nfft, np.complex128)
    sig_plan_forward = cu_fft.Plan(sig_fft_gpu.shape, np.float64, np.complex128)
    win_plan_forward = cu_fft.Plan(win_fft_gpu.shape, np.float64, np.complex128)
    cu_fft.fft(sig_gpu, sig_fft_gpu, sig_plan_forward)
    cu_fft.fft(win_gpu, win_fft_gpu, win_plan_forward)

    # Convolve
    out_fft = linalg.multiply(sig_fft_gpu, win_fft_gpu, overwrite=True)
    linalg.scale(2.0, out_fft)

    # Plan inverse
    out_gpu = gpuarray.empty_like(out_fft)
    plan_inverse = cu_fft.Plan(out_fft.shape, np.complex128, np.complex128)
    cu_fft.ifft(out_fft, out_gpu, plan_inverse, True)
    out_np = np.zeros(len(out_gpu), np.complex128)
    out_gpu.get(out_np)
    return out_np


def pointwise_np(signal: np.array, window: np.array):
    return np.convolve(signal, window).astype(np.complex128)


if __name__ == "__main__":
    fname = sys.argv[2]
    dirout = sys.argv[3]
    if int(sys.argv[1]) == 1:
        load_direct = True
    else:
        load_direct = False
        with open(fname, 'r') as fp:
            config = json.load(fp)

    methods = {'numpy_fft': fftea_time_np, 'fftw': fftea_time_fftw, 'cufft': fftea_time_cuda,
               'numpy_pointwise': pointwise_np}

    for m_name, func in methods.items():
        if load_direct:
            container = np.load(fname)
            sig = container['sig']
            win = container['win']
            name = str(container['name'])
            outname = dirout + name + '_' + str(m_name) + '.out'
        else:
            sig = gen_sig(config['frequencies'], config['sig_len'], config['sampling_rate'])
            win = gen_window(config['win_len'], config['cutoff_freq'], config['sampling_rate'])
            if 'noise' in config.keys():
                sig = add_noise(sig, config['noise']['mean'], config['noise']['std'], config['noise']['frequency'],
                                config['sampling_rate'], config['noise']['seed'], config['noise']['alpha'])
            outname = dirout + config['name'] + '_' + str(m_name) + '.out'

        result = func(sig, win)
        np.save(outname, result)
