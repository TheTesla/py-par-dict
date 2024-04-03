#!/usr/bin/env python3
import numpy as np
import numba as nb
from numba import njit, prange
import time

#@njit(parallel=True,nogil=True)
@njit(parallel=True)
def new_par_dict(key_type, val_type, nothrds=4, fifo_size=1024):
    keys = np.zeros((nothrds,nothrds,fifo_size),dtype=key_type)
    vals = np.zeros((nothrds,nothrds,fifo_size),dtype=val_type)
    cmds = np.zeros((nothrds,nothrds,fifo_size),dtype=np.int8)
    rd_idx = np.zeros((nothrds,nothrds),dtype=np.int64)
    wr_idx = np.zeros((nothrds,nothrds),dtype=np.int64)
    dicts = [nb.typed.Dict.empty(key_type=key_type, value_type=val_type) \
							for e in range(nothrds)]
    return (keys,vals,cmds,rd_idx,wr_idx,fifo_size,nothrds,dicts)

#@njit(nogil=True)
@njit
def par_dict_setitem(state, key, val, thrd_id=None):
    if thrd_id is None:
        thrd_id = nb.get_thread_id()
    thrd_id = int(thrd_id)

    keys, vals, cmds, rd_idx, wr_idx, fifo_size, nothrds, dicts = state


    # write into FIFO
    dst_id = hash(key)%nothrds
    src_id = thrd_id
    # check FIFO overflow
    cap = (rd_idx[src_id, dst_id] - wr_idx[src_id, dst_id] - 1)%fifo_size
    if cap == 0:
        print("FIFO overflow!")
    wridx = wr_idx[src_id, dst_id]
    keys[src_id, dst_id, wridx] = key
    vals[src_id, dst_id, wridx] = val
    cmds[src_id, dst_id, wridx] = True
    wr_idx[src_id, dst_id] = (wr_idx[src_id, dst_id] + 1)%fifo_size

    # read FIFO into dicts
    dst_id = thrd_id
    if hash(key)%1000 == 0:
        for src_id in range(nothrds):
            while wr_idx[src_id, dst_id] != rd_idx[src_id, dst_id]:
                k = keys[src_id, dst_id, rd_idx[src_id, dst_id]]
                v = vals[src_id, dst_id, rd_idx[src_id, dst_id]]
                c = cmds[src_id, dst_id, rd_idx[src_id, dst_id]]
                rd_idx[src_id, dst_id] = (rd_idx[src_id, dst_id] + 1)%fifo_size
                if c:
                    dicts[dst_id][k] = v
                else:
                    del dicts[dst_id][k]

#@njit(nogil=True)
@njit
def par_dict_delitem(state, key, thrd_id=None):
    if thrd_id is None:
        thrd_id = nb.get_thread_id()
    thrd_id = int(thrd_id)

    keys, vals, cmds, rd_idx, wr_idx, fifo_size, nothrds, dicts = state


    # write into FIFO
    dst_id = hash(key)%nothrds
    src_id = thrd_id
    # check FIFO overflow
    cap = (rd_idx[src_id, dst_id] - wr_idx[src_id, dst_id] - 1)%fifo_size
    if cap == 0:
        print("FIFO overflow!")
    wridx = wr_idx[src_id, dst_id]
    keys[src_id, dst_id, wridx] = key
    cmds[src_id, dst_id, wridx] = False
    wr_idx[src_id, dst_id] = (wr_idx[src_id, dst_id] + 1)%fifo_size

    # read FIFO into dicts
    dst_id = thrd_id
    if hash(key)%1000 == 0:
        for src_id in range(nothrds):
            while wr_idx[src_id, dst_id] != rd_idx[src_id, dst_id]:
                k = keys[src_id, dst_id, rd_idx[src_id, dst_id]]
                v = vals[src_id, dst_id, rd_idx[src_id, dst_id]]
                c = cmds[src_id, dst_id, rd_idx[src_id, dst_id]]
                rd_idx[src_id, dst_id] = (rd_idx[src_id, dst_id] + 1)%fifo_size
                if c:
                    dicts[dst_id][k] = v
                else:
                    del dicts[dst_id][k]

#@njit(parallel=True,nogil=True)
@njit(parallel=True)
def par_dict_sync(state):
    keys, vals, cmds, rd_idx, wr_idx, fifo_size, nothrds, dicts = state
    for dst_id in prange(nothrds):
        for src_id in range(nothrds):
            while wr_idx[src_id, dst_id] != rd_idx[src_id, dst_id]:
                k = keys[src_id, dst_id, rd_idx[src_id, dst_id]]
                v = vals[src_id, dst_id, rd_idx[src_id, dst_id]]
                c = cmds[src_id, dst_id, rd_idx[src_id, dst_id]]
                rd_idx[src_id, dst_id] = (rd_idx[src_id, dst_id] + 1)%fifo_size
                if c:
                    dicts[dst_id][k] = v
                else:
                    del dicts[dst_id][k]


#@njit(nogil=True)
@njit
def par_dict_getitem(state, key):
    keys, vals, cmds, rd_idx, wr_idx, fifo_size, nothrds, dicts = state
    return dicts[hash(key)%nothrds][key]

#@njit(parallel=True,nogil=True)
@njit(parallel=True)
def demo():
    n = 8000000
    no_threads = nb.get_num_threads()
    #no_threads = 8
    pdict = new_par_dict(np.int64, nb.types.float64, no_threads, 2**20)
    par_dict_setitem(pdict, int(23), 42.0)

    for i in prange(n):
        k = i
        v = np.sin(8**(1/(1+i))*3.0)**0.3
        par_dict_setitem(pdict, k, v)

    print([len(e) for e in pdict[-1]])
    par_dict_sync(pdict)
    print([len(e) for e in pdict[-1]])
    #print(pdict[-1])

    tmp = 0
    for i in prange(n):
        tmp += par_dict_getitem(pdict, i)

    print(tmp)

    for i in prange(n):
        par_dict_delitem(pdict, i)

    print([len(e) for e in pdict[-1]])
    par_dict_sync(pdict)
    print([len(e) for e in pdict[-1]])

if __name__ == "__main__":
    for n in range(nb.config.NUMBA_NUM_THREADS):
        if n == 0:
            n = 1
        nb.set_num_threads(n)
        t0 = time.time()
        demo()
        t = time.time() - t0
        print(f"- threads: {n: 2} - time: {t: 3.2f} cputime: {t*n: 3.2f}")


