import matplotlib.pyplot as plt
import numpy as np


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


def determine_size(length):
    return int(2 ** np.ceil(np.log2(length)))


def fftea_time_np(freqs: list, sig_len: int, win_len: int, cut_freq: int, srate: int):
    signal = gen_sig(freqs, sig_len, srate)
    window = gen_window(win_len, cut_freq, srate)
    nfft = determine_size(len(signal) + len(window) - 1)
    sig_zero_pad = np.zeros(nfft)
    win_zero_pad = np.zeros(nfft)
    sig_zero_pad[0:len(signal)] = signal
    win_zero_pad[0:len(window)] = window
    sig_fft = np.fft.fft(sig_zero_pad)
    win_fft = np.fft.fft(win_zero_pad)
    out_fft = np.multiply(sig_fft, win_fft)
    out = np.fft.ifft(out_fft)
    return np.real(out), np.imag(out)


def fftea_time_fftw(freqs: list, sig_len: int, win_len: int, cut_freq: int, srate: int):
    import pyfftw
    signal = gen_sig(freqs, sig_len, srate)
    window = gen_window(win_len, cut_freq, srate)
    nfft = determine_size(len(signal) + len(window) - 1)
    sig_zero_pad = pyfftw.empty_aligned(len(signal), dtype='float64')
    win_zero_pad = pyfftw.empty_aligned(len(window), dtype='float64')
    sig_zero_pad[0:len(signal)] = signal
    win_zero_pad[0:len(window)] = window
    sig_fft = pyfftw.interfaces.numpy_fft.fft(sig_zero_pad, n=nfft)
    win_fft = pyfftw.interfaces.numpy_fft.fft(win_zero_pad, n=nfft)
    out_fft = np.multiply(sig_fft, win_fft)
    out = pyfftw.interfaces.numpy_fft.ifft(out_fft, n=nfft)
    return np.real(out), np.imag(out)


def fftea_time_cuda(freqs: list, sig_len: int, win_len: int, cut_freq: int, srate: int):
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    import skcuda.fft as cu_fft
    import skcuda.linalg as linalg
    ctx = pycuda.autoinit.make_default_context()
    ctx.pop()
    linalg.init()
    signal = gen_sig(freqs, sig_len, srate)
    window = gen_window(win_len, cut_freq, srate)
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
    return np.real(out_np), np.imag(out_np)


def plot():
    pass


if __name__ == "__main__":
    frequencies = [440, 800, 1000, 2000]
    M = 256  # Signal size
    L = 256  # Filter size
    cutoff_freq = 600
    sampling_rate = 5000

    np_filtered, error = fftea_time_np(frequencies, M, L, cutoff_freq, sampling_rate)
    fftw_filtered, fftw_error = fftea_time_fftw(frequencies, M, L, cutoff_freq, sampling_rate)
    cuda_filtered, cuda_error = fftea_time_cuda(frequencies, M, L, cutoff_freq, sampling_rate)

    tol = 1e-15
    print(np.allclose(np_filtered, fftw_filtered, rtol=tol))
    print(np.allclose(np_filtered, cuda_filtered, rtol=tol))
    print(np.allclose(fftw_filtered, cuda_filtered, rtol=tol))
    plt.plot(np_filtered)
    plt.plot(fftw_filtered)
    plt.plot(cuda_filtered)
    plt.show()
