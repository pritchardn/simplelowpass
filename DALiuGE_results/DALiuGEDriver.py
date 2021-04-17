# (c) Nicholas Pritchard 2021
# Ensure a DALiuGE Master Manager is running on port 9000
# Ensure a DALiuGE Daemon is running on port 8000

import csv

from .labTools import full_trial_single


def main():
    methods = ['cuda_fft', 'fftw_fft', 'numpy_fft', 'numpy_pointwise']
    for method in methods:
        for i in range(0, 14):
            prefix = method + '_' + str(i)
            print(prefix)
            full_trial_single(prefix, './lowpass_graphs/', './' + prefix + '_')


def rerun():
    methods = ['cuda_fft', 'fftw_fft', 'numpy_fft', 'numpy_pointwise']
    with open('rerun.csv', 'w', newline='') as csvf:
        fieldnames = ['Method']
        for i in range(4):
            fieldnames.append(str(i))
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for method in methods:
            row = {fieldnames[0]: method}
            for i in range(4):
                prefix = method + '_' + str(i)
                fname = prefix + '_out.csv'
                hashes = csv.DictReader(open(fname))
                for line in hashes:
                    if line['Hash'] == 'ReproducibilityFlags.RERUN':
                        row[fieldnames[i + 1]] = line[prefix]
            writer.writerow(row)


def repeat():
    methods = ['cuda_fft', 'fftw_fft', 'numpy_fft', 'numpy_pointwise']
    with open('repeat.csv', 'w', newline='') as csvf:
        fieldnames = ['Method']
        for i in range(10):
            fieldnames.append(str(i))
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for method in methods:
            row = {fieldnames[0]: method}
            for i in range(10):
                prefix = method + '_' + str(i + 4)
                fname = prefix + '_out.csv'
                hashes = csv.DictReader(open(fname))
                for line in hashes:
                    if line['Hash'] == 'ReproducibilityFlags.REPEAT':
                        row[fieldnames[i + 1]] = line[prefix]
            writer.writerow(row)


def recompute():
    methods = ['cuda_fft', 'fftw_fft', 'numpy_fft', 'numpy_pointwise']
    with open('recompute.csv', 'w', newline='') as csvf:
        fieldnames = ['Method']
        for i in range(14):
            fieldnames.append(str(i))
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for method in methods:
            row = {fieldnames[0]: method}
            for i in range(14):
                prefix = method + '_' + str(i)
                fname = prefix + '_out.csv'
                hashes = csv.DictReader(open(fname))
                for line in hashes:
                    if line['Hash'] == 'ReproducibilityFlags.RECOMPUTE':
                        row[fieldnames[i + 1]] = line[prefix]
            writer.writerow(row)


def reproduce():
    methods = ['cuda_fft', 'fftw_fft', 'numpy_fft', 'numpy_pointwise']
    with open('reproduce.csv', 'w', newline='') as csvf:
        fieldnames = ['Method']
        for i in range(14):
            fieldnames.append(str(i))
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for method in methods:
            row = {fieldnames[0]: method}
            for i in range(14):
                prefix = method + '_' + str(i)
                fname = prefix + '_out.csv'
                hashes = csv.DictReader(open(fname))
                for line in hashes:
                    if line['Hash'] == 'ReproducibilityFlags.REPRODUCE':
                        row[fieldnames[i + 1]] = line[prefix]
            writer.writerow(row)


def replicate_scientific():
    methods = ['cuda_fft', 'fftw_fft', 'numpy_fft', 'numpy_pointwise']
    with open('replicate_sci.csv', 'w', newline='') as csvf:
        fieldnames = ['Method']
        for i in range(14):
            fieldnames.append(str(i))
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for method in methods:
            row = {fieldnames[0]: method}
            for i in range(14):
                prefix = method + '_' + str(i)
                fname = prefix + '_out.csv'
                hashes = csv.DictReader(open(fname))
                for line in hashes:
                    if line['Hash'] == 'ReproducibilityFlags.REPLICATE_SCI':
                        row[fieldnames[i + 1]] = line[prefix]
            writer.writerow(row)


def replicate_computational():
    methods = ['cuda_fft', 'fftw_fft', 'numpy_fft', 'numpy_pointwise']
    with open('replicate_comp.csv', 'w', newline='') as csvf:
        fieldnames = ['Method']
        for i in range(14):
            fieldnames.append(str(i))
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for method in methods:
            row = {fieldnames[0]: method}
            for i in range(14):
                prefix = method + '_' + str(i)
                fname = prefix + '_out.csv'
                hashes = csv.DictReader(open(fname))
                for line in hashes:
                    if line['Hash'] == 'ReproducibilityFlags.REPLICATE_COMP':
                        row[fieldnames[i + 1]] = line[prefix]
            writer.writerow(row)


def replicate_total():
    methods = ['cuda_fft', 'fftw_fft', 'numpy_fft', 'numpy_pointwise']
    with open('replicate_total.csv', 'w', newline='') as csvf:
        fieldnames = ['Method']
        for i in range(14):
            fieldnames.append(str(i))
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for method in methods:
            row = {fieldnames[0]: method}
            for i in range(14):
                prefix = method + '_' + str(i)
                fname = prefix + '_out.csv'
                hashes = csv.DictReader(open(fname))
                for line in hashes:
                    if line['Hash'] == 'ReproducibilityFlags.REPLICATE_TOTAL':
                        row[fieldnames[i + 1]] = line[prefix]
            writer.writerow(row)


if __name__ == "__main__":
    main()
    rerun()
    repeat()
    recompute()
    reproduce()
    replicate_scientific()
    replicate_computational()
    replicate_total()
