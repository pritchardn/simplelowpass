"""
Performs the second published reproducibility test
:argv[1]: The relative base path of single precision input files
:argv[2]: The relative base path of double precision input files
:argv[3]: The relative base path of output files
"""
import csv
import glob
import sys

import numpy as np


def normalize_signal(series):
    """
    Normalises the input series
    :param series:
    :return: The normalised input series
    """
    # amean = np.mean(series)
    astd = np.std(series)
    series /= astd
    return series


def correlate(series_a, series_b):
    """
    Computes the correlation of the two input signals
    :param series_a:
    :param series_b:
    :return: The correaltion of the two input signals
    """
    return np.absolute(np.correlate(series_a, series_b, mode='valid') / len(series_a))


def main(dir_sin, dir_doub, fout):
    """
    Computes the second published reproduction test.
    Writes a csv result file.
    :param dir_sin: The relative base path of the single precision input files
    :param dir_doub: The relative base path of the double precision input files
    :param fout: The relative base path for the output file.
    """
    methods = ['cufft', 'fftw', 'numpy_fft', 'numpy_pointwise']

    with open(fout + '.csv', 'w', newline='') as csvf:
        fieldnames = ['Precision (NCC)'] + methods
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        outs = {m: 0.0 for m in methods}
        i = 0
        while len(glob.glob(dir_sin + str(i) + '*.out.npy')) != 0:
            for method in methods:
                series_x = normalize_signal(np.load(dir_sin + str(i) + '_' + method + '.out.npy'))
                series_y = normalize_signal(np.load(dir_doub + str(i) + '_' + method + '.out.npy'))
                auto = correlate(series_y, series_y)
                comp = correlate(series_x, series_y)
                outs[method] += abs(auto - comp)
            i += 1
        row = {fieldnames[0]: 'Single vs. Double (NCC)'}
        j = 1
        for method in methods:
            outs[method] /= i
            row[fieldnames[j]] = round(outs[method][0], 15)
            j += 1
        writer.writerow(row)


if __name__ == '__main__':
    DIR_SINGLE = sys.argv[1]
    DIR_DOUBLE = sys.argv[2]
    FILE_OUT = sys.argv[3]
    main(DIR_SINGLE, DIR_DOUBLE, FILE_OUT)
