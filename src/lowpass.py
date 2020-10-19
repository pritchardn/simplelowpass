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
    xzp[0:len(signal)] = signal
    hzp[0:len(window)] = window
    X = np.fft.fft(xzp)
    H = np.fft.fft(hzp)
    Y = np.multiply(X, H)
    y = np.fft.ifft(Y)
    return np.real(y), np.imag(y)


def fftea_time_fftw(signal: np.array, window: np.array):
    nfft = determine_size(len(signal) + len(window) - 1)
    xzp = pyfftw.empty_aligned(len(signal), dtype='float64')
    hzp = pyfftw.empty_aligned(len(window), dtype='float64')
    xzp[0:len(signal)] = signal
    hzp[0:len(window)] = window
    X = pyfftw.interfaces.numpy_fft.fft(xzp, n=nfft)
    H = pyfftw.interfaces.numpy_fft.fft(hzp, n=nfft)
    Y = np.multiply(X, H)
    y = pyfftw.interfaces.numpy_fft.ifft(Y, n=nfft)
    return np.real(y), np.imag(y)


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
    xzp = np.zeros(nfft)
    hzp = np.zeros(nfft)
    xzp[0:len(signal)] = signal
    hzp[0:len(window)] = window
    sig_gpu = gpuarray.empty((nfft,), dtype=np.float64)
    win_gpu = gpuarray.empty((nfft,), dtype=np.float64)
    sig_gpu.set(xzp)
    win_gpu.set(hzp)

    # Plan forwards
    sig_fft_gpu = gpuarray.empty(nfft, np.complex128)
    win_fft_gpu = gpuarray.empty(nfft, np.complex128)
    sig_plan_forward = cu_fft.Plan(sig_fft_gpu.shape, np.float64, np.complex128)
    win_plan_forward = cu_fft.Plan(win_fft_gpu.shape, np.float64, np.complex128)
    cu_fft.fft(sig_gpu, sig_fft_gpu, sig_plan_forward)
    cu_fft.fft(win_gpu, win_fft_gpu, win_plan_forward)

    # Convolve
    y_gpu = linalg.multiply(sig_fft_gpu, win_fft_gpu, overwrite=True)
    linalg.scale(2.0, y_gpu)

    # Plan inverse
    out_gpu = gpuarray.empty_like(y_gpu)
    plan_inverse = cu_fft.Plan(y_gpu.shape, np.complex128, np.complex128)
    cu_fft.ifft(y_gpu, out_gpu, plan_inverse, True)
    out_np = np.zeros(len(out_gpu), np.complex128)
    out_gpu.get(out_np)
    return np.real(out_np), np.imag(out_np)


if __name__ == "__main__":
    frequencies = [440, 800, 1000, 2000]
    M = 256  # Signal size
    L = 256  # Filter size
    cutoff_freq = 600
    sampling_rate = 5000
    np_sig = gen_sig(frequencies, M, sampling_rate)
    np_h = window(L, cutoff_freq, sampling_rate)

    fftw_sig = gen_sig(frequencies, M, sampling_rate)
    fftw_h = window(L, cutoff_freq, sampling_rate)

    cuda_sig = gen_sig(frequencies, M, sampling_rate)
    cuda_h = window(L, cutoff_freq, sampling_rate)

    np_filtered, error = fftea_time_np(np_sig, np_h)
    fftw_filtered, fftw_error = fftea_time_fftw(fftw_sig, fftw_h)
    cuda_filtered, cuda_error = fftea_time_cuda(cuda_sig, cuda_h)

    tol = 1e-15
    print(np.allclose(np_filtered, fftw_filtered, rtol=tol))
    print(np.allclose(np_filtered, cuda_filtered, rtol=tol))
    print(np.allclose(fftw_filtered, cuda_filtered, rtol=tol))
    plt.plot(np_filtered)
    plt.plot(fftw_filtered)
    plt.plot(cuda_filtered)
    plt.show()
