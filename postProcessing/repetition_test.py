import sys
import csv
import glob
import numpy as np

if __name__ == '__main__':
    fin = sys.argv[1]
    fout = sys.argv[2]
    fpure = sys.argv[3]
    methods = ['cufft', 'fftw', 'numpy_fft', 'numpy_pointwise']
    # Load filter of a pure signal
    ground_truth = np.load(fpure)
    # Compute the normalized cross-correlation of itself (expecting 1)
    gmean = np.mean(ground_truth)
    gstd = np.std(ground_truth)
    ground_truth /= gstd
    gcorr = np.absolute(np.correlate(ground_truth, ground_truth)/len(ground_truth))
    print(gcorr)
    with open(fout + '.csv', 'w', newline='') as csvf:
        fieldnames = ['Method', 'Normalized Cross Correlation (NCC)']
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({fieldnames[0]: 'ground_truth', fieldnames[1]: gcorr})
        for method in methods:
            average = 0.0
            i = 0
            for name in glob.glob(fin + '*_' + method + '.out.npy'):
                # Compute the normalized cross-correlation of a filtered noisy signal with the filtered pure signal.
                # This gives us a measure of how effective each method is at filtering.
                print(name)
                filtered = np.load(name)
                fmean = np.mean(filtered)
                fst = np.std(filtered)
                filtered /= fst
                corr = np.absolute(np.correlate(filtered, ground_truth) / len(filtered))
                print(corr)
                average += corr
                i += 1
            writer.writerow({fieldnames[0]: method, fieldnames[1]: average/i})

