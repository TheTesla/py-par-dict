#!/usr/bin/env python3
import numpy as np
import numba as nb
from numba import njit, prange
import time


@njit(parallel=True)
#@njit(parallel=False)
def demo():
    n = 10000000
    adict = np.zeros(100000000,dtype=np.float64)
    adict[hash(23)%100000000] = 42.0

    for i in range(n):
        k = i
        v = np.sin(i)
        adict[hash(k)%100000000] = v

    #print(len(ndict))

    tmp = 0
    for i in range(n):
        tmp += adict[hash(i)%100000000]

    print(tmp)

    for i in range(n):
        adict[hash(i)%100000000] = 0.0

    #print(len(ndict))

if __name__ == "__main__":
    for n in range(1+nb.config.NUMBA_NUM_THREADS):
        if n == 0:
            n = 1
        nb.set_num_threads(n)
        t0 = time.time()
        demo()
        t = time.time() - t0
        print(f"- threads: {n: 2} - time: {t: 3.2f} cputime: {t*n: 3.2f}")




