#!/usr/bin/env python3
import numpy as np
import numba as nb
from numba import njit, prange
import time


#@njit(parallel=True,nogil=True)
@njit(parallel=False)
def demo():
    n = 100000
    ndict = {}
    ndict[23] = 42.0

    for j in range(100):
        for i in range(n):
            k = i
            v = np.sin(i)
            ndict[k] = v
    
        #print(len(ndict))
    
        tmp = 0
        for i in range(n):
            tmp += ndict[i]
    
        #print(tmp)
    
        for i in range(n):
            del ndict[i]
    
    print(len(ndict))

if __name__ == "__main__":
    n = 1
    nb.set_num_threads(n)
    t0 = time.time()
    demo()
    t = time.time() - t0
    print(f"- threads: {n: 2} - time: {t: 3.2f} cputime: {t*n: 3.2f}")
    n = 1
    nb.set_num_threads(n)
    t0 = time.time()
    demo()
    t = time.time() - t0
    print(f"- threads: {n: 2} - time: {t: 3.2f} cputime: {t*n: 3.2f}")




