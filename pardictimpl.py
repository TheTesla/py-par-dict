#!/usr/bin/env python3
import numpy as np
import numba as nb
from numba import njit, prange

@njit(parallel=True)
def new_par_dict(key_type, val_type, nothrds=4, fifo_size=1024):
    keys = np.zeros((nothrds,nothrds,fifo_size),dtype=key_type)
    vals = np.zeros((nothrds,nothrds,fifo_size),dtype=val_type)
    rd_idx = np.zeros((nothrds,nothrds),dtype=np.int64)
    wr_idx = np.zeros((nothrds,nothrds),dtype=np.int64)
    dicts = [nb.typed.Dict.empty(key_type=key_type, value_type=val_type) \
            for i in range(nothrds)]
    return (keys,vals,rd_idx,wr_idx,fifo_size,nothrds,dicts)

def par_dict_setitem(state, key, val, thrd_id=None):
    if thrd_id is None:
        thrd_id = nb.get_thread_id()
    thrd_id = int(thrd_id)

    keys, vals, rd_idx, wr_idx, fifo_size, nothrds, dicts = state


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
    wr_idx[src_id, dst_id] = (wr_idx[src_id, dst_id] + 1)%fifo_size

    # read FIFO into dicts
    dst_id = thrd_id
    for src_id in range(nothrds):
        while wr_idx[src_id, dst_id] != rd_idx[src_id, dst_id]:
            k = keys[src_id, dst_id, rd_idx[src_id, dst_id]]
            v = vals[src_id, dst_id, rd_idx[src_id, dst_id]]
            rd_idx[src_id, dst_id] = (rd_idx[src_id, dst_id] + 1)%fifo_size
            dicts[dst_id][k] = v

@njit(parallel=True)
def par_dict_sync(state):
    keys, vals, rd_idx, wr_idx, fifo_size, nothrds, dicts = state
    for dst_id in prange(nothrds):
        for src_id in range(nothrds):
            while wr_idx[src_id, dst_id] != rd_idx[src_id, dst_id]:
                k = keys[src_id, dst_id, rd_idx[src_id, dst_id]]
                v = vals[src_id, dst_id, rd_idx[src_id, dst_id]]
                rd_idx[src_id, dst_id] = (rd_idx[src_id, dst_id] + 1)%fifo_size
                dicts[dst_id][k] = v


@njit
def par_dict_getitem(state, key, val):
    keys, vals, rd_idx, wr_idx, fifo_size, nothrds, dicts = state
    return dicts[hash(key)%nothrds][key]

@njit(parallel=True)
def demo():
    n = 100000000
    no_threads = nb.get_num_threads()
    pdict = new_par_dict(np.int64, nb.types.float64, no_threads, 10240000)
    par_dict_setitem(pdict, int(23), 42.0)

    for i in prange(n):
        k = i
        v = np.sin(i*3.0)
        par_dict_setitem(pdict, k, v)

    print([len(e) for e in pdict[-1]])
    par_dict_sync(pdict)
    print([len(e) for e in pdict[-1]])
    #print(pdict[-1])




if __name__ == "__main__":
    demo()


