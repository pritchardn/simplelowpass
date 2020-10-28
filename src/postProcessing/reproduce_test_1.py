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


def find_format_length(a):
    return int(np.ceil(np.log10(len(a))))


def find_output_formatter(length):
    return "{:." + str(length) + "f}"


def main(dir_in, fout):

    methods = ['cufft', 'fftw', 'numpy_fft', 'numpy_pointwise']

    with open(fout + '.csv', 'w', newline='') as csvf:
        fieldnames = ['Method (NCC)'] + methods
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        outs = {m: {n: 0.0 for n in methods} for m in methods}
        i = 0
        format_length = 4
        while len(glob.glob(dir_in + str(i) + '*.out.npy')) != 0:
            for mi in methods:
                x = normalize_signal(np.load(dir_in + str(i) + '_' + mi + '.out.npy'))
                format_length = min(find_format_length(x), format_length)
                for mj in methods:
                    y = normalize_signal(np.load(dir_in + str(i) + '_' + mj + '.out.npy'))
                    outs[mi][mj] += correlate(x, y)
            i += 1
        out_formatter = find_output_formatter(format_length)
        for mi, results in outs.items():
            row = {fieldnames[0]: mi}
            for mj, corr in results.items():
                corr /= i
                row[mj] = out_formatter.format(corr[0])
            writer.writerow(row)


if __name__ == '__main__':
    directory_in = sys.argv[1]
    file_out = sys.argv[2]
    main(directory_in, file_out)
