"""
Assembles rerun workflow signatures for presentation
"""
import csv
import glob
import json
import sys


def main(dir_in, fout):
    """
    Performs the published rerun test given pre-existing data.
    :param dir_in: The directory containing reproducibility data
    :param fout: The file to write the results to
    :return:
    """
    methods = ['cufft', 'fftw', 'numpy_fft', 'numpy_pointwise']
    with open(fout + '.csv', 'w', newline='') as csvf:
        fieldnames = ['Method']
        for name in sorted(glob.glob(dir_in + '*_' + methods[0] + '.out.npy')):
            fieldnames.append(name.split('_')[0])
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for method in methods:
            signatures = []
            for name in sorted(glob.glob(dir_in + '*_' + method + '.out_2.json')):
                with open(name) as f:
                    signatures.append(json.load(f)['signature'])
            row = {fieldnames[0]: method}
            for i in range(len(signatures)):
                row[fieldnames[i + 1]] = signatures[i]
            writer.writerow(row)


if __name__ == '__main__':
    DIR_IN = sys.argv[1]
    FILE_OUT = sys.argv[2]
    main(DIR_IN, FILE_OUT)
