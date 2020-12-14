"""
Performs the published recompute test.
Traces the processing of any passed python command and computes a merkle hash of this information
"""

import sys
import trace
import csv
import random
import numpy as np
from merklelib import MerkleTree
from postProcessing.utils import system_summary


def recompute_command(command, fout):
    """
    Recomputes a given command, tracing the methods logged on the way.
    These methods are loaded in a MerkleTree and returns its root.
    :param command: The executed python command (string)
    :param fout: The name of the file to store the trace
    :return: A merkleroot of the trace file.
    """
    np.random.seed(42)  # Random number control
    random.seed(42)
    sys.stdout = open(fout, 'w')
    tracer = trace.Trace(trace=1,
                         count=0)
    tracer.run(command)
    sys.stdout.close()
    sys.stdout = sys.__stdout__

    log = open(fout, 'r')
    mtree = MerkleTree(log.readlines())
    mtree.append(system_summary()['signature'])
    return mtree.merkle_root


def main(command, fout, scratch_file, num_trials=5):
    """
    Recomputes a given command num_trials times storing the hash of its trace in a csv file
    :param command: The executed python command (string)
    :param fout: The filename of the output csv
    :param scratch_file: A file used to store the interim trace
    :param num_trials: The number of times the command is executed
    """
    with open(fout + '.csv', 'w', newline='') as csvf:
        fieldnames = ['Command', 'MerkleRoot']
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for _ in range(num_trials):
            root = recompute_command(command, scratch_file)
            writer.writerow({fieldnames[0]: command,
                             fieldnames[1]: root})


if __name__ == '__main__':
    COMMAND = sys.argv[1]
    RESULT_LOC = sys.argv[2]
    OUT_LOC = sys.argv[3]
    SCRATCH = sys.argv[4]
    NUM_TRIALS = int(sys.argv[5])
    main(COMMAND, OUT_LOC, SCRATCH, NUM_TRIALS)
