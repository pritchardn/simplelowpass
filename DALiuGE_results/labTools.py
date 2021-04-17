import csv
import json
import optparse

from dlg.common.reproducibility.reproducibility import ReproducibilityFlags
from dlg.translator.tool_commands import dlg_fill, dlg_unroll, dlg_partition, dlg_map, dlg_submit


def run_full_workflow(rmode: ReproducibilityFlags, workflow: str, workflow_loc='./', out_loc='./'):
    lgt = workflow_loc + workflow + ".graph"
    lg = out_loc + workflow + "LG.graph"
    pgs = out_loc + workflow + "PGS.graph"
    pgt = out_loc + workflow + "PGT.graph"
    pg = out_loc + workflow + "PG.graph"
    rg = out_loc + workflow + ".out"

    rmodes = str(rmode.value)

    parser = optparse.OptionParser()
    dlg_fill(parser, ['-L', lgt, '-R', rmodes, '-o', lg, '-f', 'newline'])
    parser = optparse.OptionParser()
    dlg_unroll(parser, ['-L', lg, '-o', pgs, '-f', 'newline'])
    parser = optparse.OptionParser()
    dlg_partition(parser, ['-P', pgs, '-o', pgt, '-f', 'newline'])
    parser = optparse.OptionParser()
    dlg_map(parser, ['-P', pgt, '-N', '127.0.0.1,127.0.0.1', '-o', pg, '-f', 'newline'])
    parser = optparse.OptionParser()
    dlg_submit(parser, ['-P', pg, '-p', '8000', '-R', '-o', rg, '-f', 'newline'])


def test_identical(w1: str, w2: str):
    f1 = open(w1 + '.out')
    f2 = open(w2 + '.out')
    r1 = json.load(f1)
    r2 = json.load(f2)
    f1.close()
    f2.close()
    h1 = r1['reprodata']['signature']
    h2 = r2['reprodata']['signature']
    return h1 == h2, h1, h2


def test_single(w1: str):
    f1 = open(w1 + '.out')
    r1 = json.load(f1)
    f1.close()
    return r1['reprodata']['signature']


def graph_trial(w1, w2, loc, flag=ReproducibilityFlags.NOTHING):
    run_full_workflow(flag, w1, loc)
    run_full_workflow(flag, w2, loc)
    result, h1, h2 = test_identical(w1, w2)
    return {'Hash': flag,
            w1: h1,
            w2: h2,
            'Match': result}


def graph_trial_single(w1, loc, flag=ReproducibilityFlags.NOTHING):
    run_full_workflow(flag, w1, loc)
    return {'Hash': flag,
            w1: test_single(w1)}


def full_trial_single(w1, loc, sav='./'):
    with open(sav + 'out.csv', 'w', newline='') as file:
        fieldnames = ['Hash', w1]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(graph_trial_single(w1, loc, ReproducibilityFlags.RERUN))
        writer.writerow(graph_trial_single(w1, loc, ReproducibilityFlags.REPEAT))
        writer.writerow(graph_trial_single(w1, loc, ReproducibilityFlags.RECOMPUTE))
        writer.writerow(graph_trial_single(w1, loc, ReproducibilityFlags.REPRODUCE))
        writer.writerow(graph_trial_single(w1, loc, ReproducibilityFlags.REPLICATE_SCI))
        writer.writerow(graph_trial_single(w1, loc, ReproducibilityFlags.REPLICATE_COMP))
        writer.writerow(graph_trial_single(w1, loc, ReproducibilityFlags.REPLICATE_TOTAL))


def full_trial(w1, w2, loc, sav='./'):
    with open(sav + 'out.csv', 'w', newline='') as file:
        fieldnames = ['Hash', w1, w2, 'Match']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(graph_trial(w1, w2, loc, ReproducibilityFlags.RERUN))
        writer.writerow(graph_trial(w1, w2, loc, ReproducibilityFlags.REPEAT))
        writer.writerow(graph_trial(w1, w2, loc, ReproducibilityFlags.RECOMPUTE))
        writer.writerow(graph_trial(w1, w2, loc, ReproducibilityFlags.REPRODUCE))
        writer.writerow(graph_trial(w1, w2, loc, ReproducibilityFlags.REPLICATE_SCI))
        writer.writerow(graph_trial(w1, w2, loc, ReproducibilityFlags.REPLICATE_COMP))
        writer.writerow(graph_trial(w1, w2, loc, ReproducibilityFlags.REPLICATE_TOTAL))


def compare_numpy_files(f1, f2):
    import numpy as np
    arr1 = np.fromfile(f1)
    arr2 = np.fromfile(f2)
    print("ARRAY 1 =======")
    print(arr1)
    print("ARRAY 2 =======")
    print(arr2)
    diffs = arr2 - arr1
    print("DIFFERENCES ==")
    print(diffs)
    print(np.allclose(arr1, arr2))


def plot_numpy_files(f1, f2):
    import numpy as np
    from matplotlib import pyplot as plt
    arr1 = np.fromfile(f1)
    arr2 = np.fromfile(f2)
    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.plot(np.real(arr1))
    axis.plot(np.real(arr2))
    axis.set_xlim(0, len(arr2))
    axis.set_ylim(-2, 2)
    axis.set(title='Filtered signal',
             ylabel='Amplitude',
             xlabel='Time (samples)')
    plt.savefig('filtered.png', dpi=300)
    plt.show()
