"""
A few utility functions for various low-pass filter trials
"""
import platform
import sys

import GPUtil
import numpy as np
import psutil
from merklelib import MerkleTree


def normalize_signal(series):
    """
    Normalises the input series
    :param series: The input series
    :return: Normalised input series
    """
    # amean = np.mean(series)
    astd = np.std(series)
    series /= astd
    return series


def correlate(series_a, series_b):
    """
    Computes the cross correlation between two series
    :param series_a:
    :param series_b:
    :return: The cross-correlation between the two input series
    """
    return np.absolute(np.correlate(series_a, series_b, mode='valid') / len(series_a))


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
