import json
import sys
import numpy as np

from lowpass import gen_sig, gen_window
if __name__ == '__main__':
    input_0 = {'frequencies': [440, 800, 1000, 2000], 'sig_len': 256, 'win_len': 256, 'cutoff_freq': 600,
               'sampling_rate': 5000, 'name': '0'}
    input_1 = {'frequencies': [440, 800, 1000, 2000], 'sig_len': 256, 'win_len': 128, 'cutoff_freq': 600,
               'sampling_rate': 5000, 'name': '1'}
    input_2 = {'frequencies': [440, 800, 1000, 2000], 'sig_len': 256, 'win_len': 256, 'cutoff_freq': 500,
               'sampling_rate': 5000, 'name': '2'}
    input_3 = {'frequencies': [440, 800, 1200, 2000], 'sig_len': 256, 'win_len': 256, 'cutoff_freq': 600,
               'sampling_rate': 5000, 'name': '3'}
    inputs = [input_0, input_1, input_2, input_3]
    file_loc = sys.argv[1]
    for i in range(len(inputs)):
        fname = file_loc + str(i) + '.in'
        with open(fname, 'w') as fp:
            json.dump(inputs[i], fp, sort_keys=True, indent=2)
        sig = gen_sig(inputs[i]['frequencies'], inputs[i]['sig_len'], inputs[i]['sampling_rate'])
        win = gen_window(inputs[i]['win_len'], inputs[i]['cutoff_freq'], inputs[i]['sampling_rate'])
        np.savez(fname, sig=sig, win=win, name=inputs[i]['name'])
