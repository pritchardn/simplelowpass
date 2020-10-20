import json
import sys

if __name__ == '__main__':
    input_1 = {'frequencies': [440, 800, 1000, 2000], 'sig_len': 256, 'win_len': 256, 'cutoff_freq': 600,
               'sampling_rate': 5000}
    input_2 = {'frequencies': [440, 800, 1000, 2000], 'sig_len': 256, 'win_len': 128, 'cutoff_freq': 600,
               'sampling_rate': 5000}
    input_3 = {'frequencies': [440, 800, 1000, 2000], 'sig_len': 256, 'win_len': 256, 'cutoff_freq': 500,
               'sampling_rate': 5000}
    inputs = [input_1, input_2, input_3]
    file_loc = sys.argv[1]
    for i in range(len(inputs)):
        with open(file_loc + str(i) + '.in', 'w') as fp:
            json.dump(inputs[i], fp, sort_keys=True, indent=2)
