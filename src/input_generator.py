import json
import sys

import numpy as np

from lowpass import gen_sig, gen_window, add_noise

if __name__ == '__main__':
    file_loc = sys.argv[1]
    num_random = int(sys.argv[2])
    inputs = [{'frequencies': [440, 800, 1000, 2000], 'sig_len': 256, 'win_len': 256, 'cutoff_freq': 600,
               'sampling_rate': 5000, 'name': '0'},
              {'frequencies': [440, 800, 1000, 2000], 'sig_len': 256, 'win_len': 128, 'cutoff_freq': 600,
               'sampling_rate': 5000, 'name': '1'},
              {'frequencies': [440, 800, 1000, 2000], 'sig_len': 256, 'win_len': 256, 'cutoff_freq': 500,
               'sampling_rate': 5000, 'name': '2'},
              {'frequencies': [440, 800, 1200, 2000], 'sig_len': 256, 'win_len': 256, 'cutoff_freq': 600,
               'sampling_rate': 5000, 'name': '3'}]
    for i in range(num_random):
        inputs.append({'frequencies': [440, 800, 1200, 2000], 'sig_len': 256, 'win_len': 256, 'cutoff_freq': 600,
                       'sampling_rate': 5000,
                       'noise': {'mean': 0, 'std': 1, 'frequency': 1200, 'seed': i, 'alpha': 0.1},
                       'name': str(i + 4)})

    for i in range(len(inputs)):
        fname = file_loc + str(i) + '.in'
        with open(fname, 'w') as fp:
            json.dump(inputs[i], fp, sort_keys=True, indent=2)
        sig = gen_sig(inputs[i]['frequencies'], inputs[i]['sig_len'], inputs[i]['sampling_rate'])
        win = gen_window(inputs[i]['win_len'], inputs[i]['cutoff_freq'], inputs[i]['sampling_rate'])
        if 'noise' in inputs[i].keys():
            sig = add_noise(sig, inputs[i]['noise']['mean'], inputs[i]['noise']['std'], inputs[i]['noise']['frequency'],
                            inputs[i]['sampling_rate'], inputs[i]['noise']['seed'], inputs[i]['noise']['alpha'])
            np.savez(fname, sig=sig, win=win, name=inputs[i]['name'], noise=True)
        else:
            np.savez(fname, sig=sig, win=win, name=inputs[i]['name'])
