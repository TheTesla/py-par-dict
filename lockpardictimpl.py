#!/usr/bin/env python3
import numpy as np
import numba as nb
from numba import njit, prange
from numba.typed import List
from numba.types import Tuple
import time

@njit(parallel=True,cache=True)
def new_par_dict(key_type, val_type, nobuckets=4):
    dicts = list([nb.typed.Dict.empty(key_type=key_type, value_type=val_type) \
                    for e in range(nobuckets)])
    #nothreads=16
    #locks = np.zeros((nothreads,nobuckets),dtype=np.uint)
    locks = np.zeros(nobuckets,dtype=np.uint)
    return (nobuckets,dicts,locks)
    #return (nobuckets,dicts)

@njit(cache=True)
def par_dict_setitem(state, key, val):
    nobuckets, dicts, locks = state
    #nobuckets, dicts = state
    thrd_id = nb.get_thread_id()

    lock = 255
    while lock:
        lock = locks[hash(key)%nobuckets]
        locks[hash(key)%nobuckets] = thrd_id + 1
        if lock != locks[hash(key)%nobuckets]:
            lock = 0

    dicts[hash(key)%nobuckets][key] = val

    locks[hash(key)%nobuckets] = 0

@njit(cache=True)
def par_dict_delitem(state, key):
    nobuckets, dicts, locks = state
    #nobuckets, dicts = state
    thrd_id = nb.get_thread_id()
    lock = 255
    while lock:
        lock = locks[hash(key)%nobuckets]
        locks[hash(key)%nobuckets] = thrd_id + 1
        if lock != locks[hash(key)%nobuckets]:
            lock = 0

    del dicts[hash(key)%nobuckets][key]

    locks[hash(key)%nobuckets] = 0

@njit
def par_dict_getitem(state, key):
    nobuckets, dicts, locks = state
    #nobuckets, dicts = state
    return dicts[hash(key)%nobuckets][key]

@njit(parallel=True)
def demo():
    n = 10000
    no_threads = nb.get_num_threads()
    pdict = new_par_dict(np.int64, nb.types.float64, 2000)

    for m in range(1000):
        par_dict_setitem(pdict, int(23), 42.0)
        for i in prange(n):
            k = i
            v = np.sin(i)
            par_dict_setitem(pdict, k, v)

        tmp = 0
        nn = n
        for i in prange(nn):
            tmp += par_dict_getitem(pdict, i)
            #tmp += i

        #print(tmp)

        for i in prange(n):
            par_dict_delitem(pdict, i)

        #print([len(e) for e in pdict[-1]])

if __name__ == "__main__":
    for n in range(1+nb.config.NUMBA_NUM_THREADS):
        if n == 0:
            n = 1
        nb.set_num_threads(n)
        t0 = time.time()
        demo()
        t = time.time() - t0
        print(f"- threads: {n: 2} - time: {t: 3.2f} cputime: {t*n: 3.2f}")


