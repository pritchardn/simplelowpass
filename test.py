import numpy as np
import glob


def normalize_signal(a):
    amean = np.mean(a)
    astd = np.std(a)
    a /= astd
    return a


def correlate(a, b):
    return np.absolute(np.correlate(a, b, mode='valid') / len(a))


if __name__ == '__main__':
    orig = normalize_signal(np.load('./data/3.in.npz')['sig'])
    print(correlate(orig, orig))
    for i in range(4, 14):
        new = normalize_signal(np.load('./data/'+str(i)+'.in.npz')['sig'])
        print(correlate(new, orig))
