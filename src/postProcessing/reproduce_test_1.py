"""
Performs the first published reproducibility test
:argv[1]: The relative base path of input files
:argv[2]: The relative base path of output files
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


def find_format_length(series):
    """
    Finds the number of significant figures needed for the input series
    :param series:
    :return: The number of significant figures needed for normalised cross-correlation accuracy
    """
    return int(np.ceil(np.log10(len(series))))


def find_output_formatter(length):
    """
    Finds the printing formatter for the number of significant figures required
    :param length:
    :return: A format specifier to be used with Python's default printing function
    """
    return "{:." + str(length) + "f}"


def main(dir_in, fout):
    """
    Computes the first published reproduction test.
    Writes a csv result file.
    :param dir_in: The relative base path of the input files
    :param fout: The relative base path for the output file.
    """
    methods = ['cufft', 'fftw', 'numpy_fft', 'numpy_pointwise']

    with open(fout + '.csv', 'w', newline='') as csvf:
        fieldnames = ['Method (NCC)'] + methods
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        outs = {m: {n: 0.0 for n in methods} for m in methods}
        i = 0
        format_length = 4
        while len(glob.glob(dir_in + str(i) + '*.out.npy')) != 0:
            for method_1 in methods:
                series_x = normalize_signal(np.load(dir_in + str(i) + '_' + method_1 + '.out.npy'))
                format_length = min(find_format_length(series_x), format_length)
                for method_2 in methods:
                    series_y = normalize_signal(np.load(dir_in + str(i) + '_' + method_2 + '.out.npy'))
                    outs[method_1][method_2] += correlate(series_x, series_y)
            i += 1
        out_formatter = find_output_formatter(format_length)
        for method_1, results in outs.items():
            row = {fieldnames[0]: method_1}
            for method_2, corr in results.items():
                corr /= i
                row[method_2] = out_formatter.format(corr[0])
            writer.writerow(row)


if __name__ == '__main__':
    DIRECTORY_IN = sys.argv[1]
    FILE_OUT = sys.argv[2]
    main(DIRECTORY_IN, FILE_OUT)
