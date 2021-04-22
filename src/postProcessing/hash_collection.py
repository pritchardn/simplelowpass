import csv
import glob
import json

from reproducibility import ReproducibilityFlags


def collect(dir_in, fout, rmode: ReproducibilityFlags):
    """
    Performs the published rerun test given pre-existing data.
    :param dir_in: The directory containing reproducibility data
    :param fout: The file to write the results to
    :param rmode: The particular standard to summarise mode
    :return:
    """
    methods = ['cufft', 'fftw', 'numpy_fft', 'numpy_pointwise']
    with open(fout + '.csv', 'w', newline='') as csvf:
        fieldnames = ['Method']
        for name in sorted(glob.glob(dir_in + '/**/' + '*_' + methods[0] + '.out.npy', recursive=True)):
            fieldnames.append(name.split('_')[0])
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for method in methods:
            signatures = []
            for name in sorted(
                    glob.glob(dir_in + '/**/' + '*_' + method + '.out_' + str(rmode.value) + '.json', recursive=True)):
                with open(name) as f:
                    signatures.append(json.load(f)['signature'])
            row = {fieldnames[0]: method}
            for i in range(len(signatures)):
                row[fieldnames[i + 1]] = signatures[i]
            writer.writerow(row)
