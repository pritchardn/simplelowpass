import numpy as np
from postProcessing.utils import normalize_signal, correlate


if __name__ == '__main__':
    orig = normalize_signal(np.load('./data/3.in.npz')['sig'])
    print(correlate(orig, orig))
    for i in range(4, 14):
        new = normalize_signal(np.load('./data/'+str(i)+'.in.npz')['sig'])
        print(correlate(new, orig))
