import numpy as np


def normalize_signal(a):
    amean = np.mean(a)
    astd = np.std(a)
    a /= astd
    return a


def correlate(a, b):
    print(str(len(a)) + "  " + str(len(b)))
    return np.absolute(np.correlate(a, b, mode='valid') / len(a))
