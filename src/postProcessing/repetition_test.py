import csv
import glob
import sys

import numpy as np


def normalize_signal(a):
    amean = np.mean(a)
    astd = np.std(a)
    a /= astd
    return a


def correlate(a, b):
    return np.absolute(np.correlate(a, b, mode='valid') / len(a))


def main(dir_in, fout, fpure):

    methods = ['cufft', 'fftw', 'numpy_fft', 'numpy_pointwise']
    # Load filter of a pure signal
    ground_truth = normalize_signal(np.load(fpure))
    # Compute the normalized cross-correlation of itself (expecting 1)
    gcorr = correlate(ground_truth, ground_truth)
    with open(fout + '.csv', 'w', newline='') as csvf:
        fieldnames = ['Method', 'Normalized Cross Correlation (NCC)']
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({fieldnames[0]: 'ground_truth', fieldnames[1]: gcorr[0]})
        for method in methods:
            average = 0.0
            i = 0
            for name in glob.glob(dir_in + '*_' + method + '.out.npy'):
                # Compute the normalized cross-correlation of a filtered noisy signal with the filtered pure signal.
                # This gives us a measure of how effective each method is at filtering.
                filtered = normalize_signal(np.load(name))
                corr = correlate(filtered, ground_truth)
                average += corr
                i += 1
            writer.writerow({fieldnames[0]: method, fieldnames[1]: (average / i)[0]})


if __name__ == '__main__':
    directory_in = sys.argv[1]
    file_out = sys.argv[2]
    file_pure = sys.argv[3]
    main(directory_in, file_out, file_pure)
