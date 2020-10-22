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


if __name__ == '__main__':
    dir_sin = sys.argv[1]
    dir_doub = sys.argv[2]
    fout = sys.argv[3]
    methods = ['cufft', 'fftw', 'numpy_fft', 'numpy_pointwise']
    precision = ['single', 'double']

    with open(fout + '.csv', 'w', newline='') as csvf:
        fieldnames = ['Precision (NCC)'] + methods
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        outs = {m: 0.0 for m in methods}
        i = 0
        while len(glob.glob(dir_sin + str(i) + '*.out.npy')) != 0:
            for mi in methods:
                x = normalize_signal(np.load(dir_sin + str(i) + '_' + mi + '.out.npy'))
                y = normalize_signal(np.load(dir_doub + str(i) + '_' + mi + '.out.npy'))
                outs[mi] += correlate(x, y)
            i += 1
        row = {fieldnames[0]: 'Single vs. Double (NCC)'}
        j = 1
        for m in methods:
            outs[m] /= i
            row[fieldnames[j]] = outs[m][0]
            j += 1
        writer.writerow(row)
