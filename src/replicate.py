import glob
import os
import sys

from postProcessing.repetition_test import main as repeat
from postProcessing.reproduce_test_1 import main as repro1
from postProcessing.reproduce_test_2 import main as repro2


def system_summary():
    import psutil
    import GPUtil
    import platform
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
    return system_info


def make_dirs(base):
    print(base + 'data/')
    precision = ['double', 'single']
    quality = ['raw', 'config']
    data_type = ['noisy', 'clean']
    for p in precision:
        path = base + 'results/' + p + '/'
        for q in quality:
            path2 = path + q + '/'
            for d in data_type:
                path3 = path2 + d + '/'
                print(path3)
                try:
                    os.makedirs(path3)
                except OSError as err:
                    print(err)


def process_conf(loc, results):
    from lowpass import main as lowp
    files = glob.glob(loc + '*.in')
    for f in files:
        print(f)
        lowp(f, results + 'single/config/', False, 0)
        lowp(f, results + 'double/config/', False, 1)


def process_direct(loc, results):
    from lowpass import main as lowp
    files = glob.glob(loc + "*.npz")
    for f in files:
        print(f)
        lowp(f, results + 'single/raw/', True, 0)
        lowp(f, results + 'double/raw/', True, 1)


def main(base_loc):
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
    print(system_summary())


if __name__ == "__main__":
    result_loc = sys.argv[1]
    main(result_loc)
