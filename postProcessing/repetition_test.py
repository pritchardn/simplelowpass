import csv
import glob
import sys

import numpy as np

from postProcessing.utils import normalize_signal, correlate

if __name__ == '__main__':
    dir_in = sys.argv[1]
    fout = sys.argv[2]
    fpure = sys.argv[3]
    methods = ['cufft', 'fftw', 'numpy_fft', 'numpy_pointwise']
    # Load filter of a pure signal
    ground_truth = normalize_signal(np.load(fpure))
    # Compute the normalized cross-correlation of itself (expecting 1)
    gcorr = correlate(ground_truth, ground_truth)
    print(gcorr)
    with open(fout + '.csv', 'w', newline='') as csvf:
        fieldnames = ['Method', 'Normalized Cross Correlation (NCC)']
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({fieldnames[0]: 'ground_truth', fieldnames[1]: gcorr})
        for method in methods:
            average = 0.0
            i = 0
            for name in glob.glob(dir_in + '*_' + method + '.out.npy'):
                # Compute the normalized cross-correlation of a filtered noisy signal with the filtered pure signal.
                # This gives us a measure of how effective each method is at filtering.
                print(name)
                filtered = normalize_signal(np.load(name))
                corr = correlate(filtered, ground_truth)
                print(corr)
                average += corr
                i += 1
            writer.writerow({fieldnames[0]: method, fieldnames[1]: average / i})
