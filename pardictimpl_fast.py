#!/usr/bin/env python3
import numpy as np
import numba as nb
from numba import njit, prange
from numba.typed import List
from numba.types import Tuple
import time

@njit(parallel=True,cache=True)
def new_par_dict(key_type, val_type, nothrds=4, fifo_size=1024):
    keys = np.zeros((nothrds,nothrds,fifo_size),dtype=key_type)
    vals = np.zeros((nothrds,nothrds,fifo_size),dtype=val_type)
    cmds = np.zeros((nothrds,nothrds,fifo_size),dtype=np.int8)
    rd_idx = np.zeros((nothrds,nothrds),dtype=np.int64)
    wr_idx = np.zeros((nothrds,nothrds),dtype=np.int64)
    dicts = list([nb.typed.Dict.empty(key_type=key_type, value_type=val_type) \
                    for e in range(nothrds)])
    return (keys,vals,cmds,rd_idx,wr_idx,fifo_size,nothrds,dicts)

@njit(cache=True)
def par_dict_setitem(state, key, val, thrd_id=None):
    if thrd_id is None:
        thrd_id = nb.get_thread_id()
    thrd_id = int(thrd_id)

    keys, vals, cmds, rd_idx, wr_idx, fifo_size, nothrds, dicts = state


    hkey = hash(key)
    # write into FIFO
    dst_id = hkey%nothrds
    src_id = thrd_id
    # check FIFO overflow
    cap = (rd_idx[src_id, dst_id] - wr_idx[src_id, dst_id] - 1)%fifo_size
    if cap == 0:
        print("FIFO overflow!")
    wridx = wr_idx[src_id, dst_id]
    keys[src_id, dst_id, wridx] = key
    vals[src_id, dst_id, wridx] = val
    cmds[src_id, dst_id, wridx] = 1
    wr_idx[src_id, dst_id] = (wr_idx[src_id, dst_id] + 1)%fifo_size

    # read FIFO into dicts
    if cap < 100:
        print("write")
#        dst_id = thrd_id
#        dst_dict = dicts[dst_id]
#        for src_id in range(nothrds):
#            while wr_idx[src_id, dst_id] != rd_idx[src_id, dst_id]:
#                k = keys[src_id, dst_id, rd_idx[src_id, dst_id]]
#                v = vals[src_id, dst_id, rd_idx[src_id, dst_id]]
#                c = cmds[src_id, dst_id, rd_idx[src_id, dst_id]]
#                rd_idx[src_id, dst_id] = (rd_idx[src_id, dst_id] + 1)%fifo_size
#                if c:
#                    dst_dict[k] = v
#                else:
#                    del dst_dict[k]

@njit(cache=True)
def par_dict_delitem(state, key, thrd_id=None):
    if thrd_id is None:
        thrd_id = nb.get_thread_id()
    thrd_id = int(thrd_id)

    keys, vals, cmds, rd_idx, wr_idx, fifo_size, nothrds, dicts = state


    hkey = hash(key)
    # write into FIFO
    dst_id = hkey%nothrds
    src_id = thrd_id
    # check FIFO overflow
    cap = (rd_idx[src_id, dst_id] - wr_idx[src_id, dst_id] - 1)%fifo_size
    if cap == 0:
        print("FIFO overflow!")
    wridx = wr_idx[src_id, dst_id]
    keys[src_id, dst_id, wridx] = key
    cmds[src_id, dst_id, wridx] = 0
    wr_idx[src_id, dst_id] = (wr_idx[src_id, dst_id] + 1)%fifo_size

    # read FIFO into dicts
    if cap < 100:
        print("write")
#        dst_id = thrd_id
#        dst_dict = dicts[dst_id]
#        for src_id in range(nothrds):
#            while wr_idx[src_id, dst_id] != rd_idx[src_id, dst_id]:
#                k = keys[src_id, dst_id, rd_idx[src_id, dst_id]]
#                v = vals[src_id, dst_id, rd_idx[src_id, dst_id]]
#                c = cmds[src_id, dst_id, rd_idx[src_id, dst_id]]
#                rd_idx[src_id, dst_id] = (rd_idx[src_id, dst_id] + 1)%fifo_size
#                if c:
#                    dst_dict[k] = v
#                else:
#                    del dst_dict[k]

@njit(parallel=True)
def par_dict_sync(state):
    keys, vals, cmds, rd_idx, wr_idx, fifo_size, nothrds, dicts = state
    for dst_id in prange(nothrds):
        dst_dict = dicts[dst_id]
        for src_id in range(nothrds):
            while wr_idx[src_id, dst_id] != rd_idx[src_id, dst_id]:
                k = keys[src_id, dst_id, rd_idx[src_id, dst_id]]
                v = vals[src_id, dst_id, rd_idx[src_id, dst_id]]
                c = cmds[src_id, dst_id, rd_idx[src_id, dst_id]]
                rd_idx[src_id, dst_id] = (rd_idx[src_id, dst_id] + 1)%fifo_size
                if c:
                    dst_dict[k] = v
                else:
                    del dst_dict[k]


@njit
def par_dict_getitem(state, key):
    keys, vals, cmds, rd_idx, wr_idx, fifo_size, nothrds, dicts = state
    return dicts[hash(key)%nothrds][key]

@njit(parallel=True)
def demo():
    n = 100000
    no_threads = nb.get_num_threads()
    pdict = new_par_dict(np.int64, nb.types.float64, no_threads, 2**17)

    for m in range(100):
        par_dict_setitem(pdict, int(23), 42.0)
        for i in prange(n):
            k = i
            v = np.sin(i)
            par_dict_setitem(pdict, k, v)

        par_dict_sync(pdict)

        tmp = 0
        nn = n
        for i in prange(nn):
            tmp += par_dict_getitem(pdict, i)


        for i in prange(n):
            par_dict_delitem(pdict, i)

        par_dict_sync(pdict)

if __name__ == "__main__":
    for n in range(1+nb.config.NUMBA_NUM_THREADS):
        if n == 0:
            n = 1
        nb.set_num_threads(n)
        t0 = time.time()
        demo()
        t = time.time() - t0
        print(f"- threads: {n: 2} - time: {t: 3.3f} cputime: {t*n: 3.2f}")


