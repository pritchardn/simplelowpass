"""
Performs the published repetition test.
Computes the Normalised Cross-Correlation between each noisy trial and ground truth.
"""
import csv
import glob
import sys

import numpy as np


def normalize_signal(series):
    """
    Normalises the input series
    :param series:
    :return: Normalised input series
    """
    # amean = np.mean(a)
    astd = np.std(series)
    series /= astd
    return series


def correlate(series_a, series_b):
    """
    Computes the correlation between two series
    :param series_a:
    :param series_b:
    :return: The NCC of the two-series
    """
    return np.absolute(np.correlate(series_a, series_b, mode='valid') / len(series_a))


def find_output_formatter(series):
    """
    Finds the number of significant places needed to represent full accuracy.
    :param series: The input data length
    :return: An output specifier for use with Python's print function
    """
    length = int(np.ceil(np.log10(len(series))))
    return "{:." + str(length) + "f}"


def main(dir_in, fout, fpure):
    """
    Performs the published repetition test.
    :param dir_in: The relative base path
    :param fout: The relative output path
    :param fpure: The relative base path for ground-truth data-file location.
    :return:
    """
    methods = ['cufft', 'fftw', 'numpy_fft', 'numpy_pointwise']
    # Load filter of a pure signal
    ground_truth = normalize_signal(np.load(fpure))
    # Compute the normalized cross-correlation of itself (expecting 1)
    gcorr = correlate(ground_truth, ground_truth)
    out_formatter = find_output_formatter(ground_truth)
    with open(fout + '.csv', 'w', newline='') as csvf:
        fieldnames = ['Method', 'Normalized Cross Correlation (NCC)']
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({fieldnames[0]: 'ground_truth',
                         fieldnames[1]: out_formatter.format(gcorr[0])})
        for method in methods:
            average = 0.0
            i = 0
            for name in glob.glob(dir_in + '*_' + method + '.out.npy'):
                # Compute the normalized cross-correlation of a filtered noisy
                # signal with the filtered pure signal.
                # This gives us a measure of how effective each method is at filtering.
                corr = correlate(normalize_signal(np.load(name)), ground_truth)
                average += corr
                i += 1
            writer.writerow({fieldnames[0]: method,
                             fieldnames[1]: out_formatter.format((average / i)[0])})


if __name__ == '__main__':
    DIR_IN = sys.argv[1]
    FILE_OUT = sys.argv[2]
    FILE_PURE = sys.argv[3]
    main(DIR_IN, FILE_OUT, FILE_PURE)
