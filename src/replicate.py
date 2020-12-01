"""
Replicates all published reproducibility tests.
:argv[1]: relative path of the new computed results
:argv[2]: relative path of the published results
"""
import csv
import glob
import json
import os
import sys
import platform
import psutil
import GPUtil

from merklelib import MerkleTree

from lowpass import main as lowp
from postProcessing.repetition_test import main as repeat
from postProcessing.reproduce_test_1 import main as repro1
from postProcessing.reproduce_test_2 import main as repro2


def find_loaded_modules():
    """
    :return: A list of all loaded modules
    """
    loaded_mods = []
    for name, module in sorted(sys.modules.items()):
        if hasattr(module, '__version__'):
            loaded_mods.append(name + " " + str(module.__version__))
        else:
            loaded_mods.append(name)
    return loaded_mods


def system_summary():
    """
    Summarises the system this function is run on.
    Includes system, cpu, gpu and module details
    :return: A dictionary of system details
    """
    merkletree = MerkleTree()
    system_info = {}
    uname = platform.uname()
    system_info['system'] = {
        'system': uname.system,
        'release': uname.release,
        'version': uname.version,
        'machine': uname.machine,
        'processor': uname.processor
    }
    cpu_freq = psutil.cpu_freq()
    system_info['cpu'] = {
        'cores_phys': psutil.cpu_count(logical=False),
        'cores_logic': psutil.cpu_count(logical=True),
        'max_frequency': cpu_freq.max,
        'min_frequency': cpu_freq.min
    }
    sys_mem = psutil.virtual_memory()
    system_info['memory'] = {
        'total': sys_mem.total
    }
    gpus = GPUtil.getGPUs()
    system_info['gpu'] = {}
    for gpu in gpus:
        system_info['gpu'][gpu.id] = {
            'name': gpu.name,
            'memory': gpu.memoryTotal
        }
    system_info['modules'] = find_loaded_modules()
    merkletree.append([system_info[item] for item in system_info])
    system_info['signature'] = merkletree.merkle_root
    return system_info


def make_dirs(base):
    """
    Makes directories for all trial runs
    :param base: The relative base path
    """
    print(base + 'data/')
    precision = ['double', 'single']
    quality = ['raw', 'config']
    data_type = ['noisy', 'clean']
    for prec in precision:
        path = base + 'results/' + prec + '/'
        for qual in quality:
            path2 = path + qual + '/'
            for d_type in data_type:
                path3 = path2 + d_type + '/'
                print(path3)
                try:
                    os.makedirs(path3)
                except OSError as err:
                    print(err)


def process_conf(loc, results):
    """
    Processes all data trials specified by provided configuration files.
    Expects a double || single /config file structure.
    :param loc: The relative path location
    :param results: The relative output path location
    """
    files = glob.glob(loc + '*.in')
    for file in files:
        print(file)
        lowp(file, results + 'single/config/', False, 0)
        lowp(file, results + 'double/config/', False, 1)


def process_direct(loc, results):
    """
    Processes all data trials specified by raw data files.
    :param loc: The relative input path location
    :param results: The relative output path location
    """
    files = glob.glob(loc + "*.npz")
    for file in files:
        print(file)
        lowp(file, results + 'single/raw/', True, 0)
        lowp(file, results + 'double/raw/', True, 1)


def compare_files(file_1, file_2):
    """
    Compares the outputs in two files
    :param file_1:
    :param file_2:
    :return: A boolean true if both files match contents, false otherwise
    """
    ret = True
    with open(file_1, 'r') as file1:
        lines1 = file1.readlines()
    with open(file_2, 'r') as file2:
        lines2 = file2.readlines()
    if len(lines1) != len(lines2):
        print("Number of lines do not match")
        return False
    for i in range(min(len(lines1), len(lines2))):
        if lines1[i] != lines2[i]:
            print(lines1[i] + " does not match " + lines2[i])
            ret = False
    return ret


def replicate(base_loc, pub_loc):
    """
    Runs all published tests and compares them to the orginal published results.
    :param base_loc: The relative base path for output files
    :param pub_loc: The relative base path for published files
    :return: Prints the status of replication attempts
    """
    scientific_rep = True
    computational_rep = True
    with open(base_loc + 'results/replicate.csv', 'w') as csvf:
        fieldnames = ['Test', 'Pass']
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        if compare_files(base_loc + 'results/repeat.csv', pub_loc + 'repeat.csv'):
            writer.writerow({'Test': 'Repeat', 'Pass': True})
        else:
            writer.writerow({'Test': 'Repeat', 'Pass': False})
            scientific_rep = False
            print("Repeat fails")
        if compare_files(base_loc + 'results/reproduce1.csv', pub_loc + 'reproduce1.csv'):
            writer.writerow({'Test': 'Reproduce 1', 'Pass': True})
        else:
            writer.writerow({'Test': 'Reproduce 1', 'Pass': False})
            scientific_rep = False
            print("Reproduce 1 fails")
        if compare_files(base_loc + 'results/reproduce2.csv', pub_loc + 'reproduce2.csv'):
            writer.writerow({'Test': 'Reproduce 2', 'Pass': True})
        else:
            writer.writerow({'Test': 'Reproduce 2', 'Pass': False})
            scientific_rep = False
            print("Reproduce 2 fails")
        if not compare_files(base_loc + 'results/system.json', pub_loc + 'system.json'):
            computational_rep = False
            print("Machines differ")
        if scientific_rep:
            writer.writerow({'Test': 'Scientific Replicate', 'Pass': True})
            if computational_rep:
                writer.writerow({'Test': 'Computational Replicate', 'Pass': True})
            else:
                writer.writerow({'Test': 'Computational Replicate', 'Pass': False})
        else:
            writer.writerow({'Test': 'Scientific Replicate', 'Pass': False})
            writer.writerow({'Test': 'Computational Replicate', 'Pass': False})
    print(scientific_rep)
    print(computational_rep)


def main(base_loc, pub_loc):
    """
    Sets up and runs all published tests and compares them to the original published results.
    :param base_loc: The relative base path for output files
    :param pub_loc: The relative base path for published files
    :return: Prints status updates
    """
    print("Creating result files")
    make_dirs(base_loc)
    print("Processing from config")
    process_conf(base_loc + 'data/', base_loc + 'results/')
    print("Processing from direct files")
    process_direct(base_loc + 'data/', base_loc + 'results/')
    print("Repetition Analysis")
    repeat(base_loc + 'results/double/raw/clean/', base_loc + "results/repeat",
           base_loc + 'results/double/raw/clean/3_numpy_pointwise.out.npy')
    print("Reproduction Analysis 1")
    repro1(base_loc + 'results/double/raw/clean/', base_loc + 'results/reproduce1')
    print("Reproduction Analysis 2")
    repro2(base_loc + 'results/single/raw/clean/', base_loc + 'results/double/raw/clean/',
           base_loc + 'results/reproduce2')
    sys_summ = system_summary()
    with open(base_loc + 'results/system.json', 'w') as file:
        json.dump(sys_summ, file)
    replicate(base_loc, pub_loc)


if __name__ == "__main__":
    RESULT_LOC = sys.argv[1]
    PUBLISHED_LOC = sys.argv[2]
    main(RESULT_LOC, PUBLISHED_LOC)
