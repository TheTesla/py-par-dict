from numba import njit, prange
from numba.typed import List, Dict
import numba as nb
import numpy as np

@njit(parallel=True)
def par(x):
    print(nb.get_num_threads())
    print(nb.get_thread_id())


    # initialize parDict
    n = nb.get_num_threads()
    fifo_cap = 1024
    fifos_k = np.zeros((n,n,fifo_cap),dtype=np.int64)
    fifos_v = np.zeros((n,n,fifo_cap),dtype=np.float64)
    fifo_idx = np.zeros((n,n,2),dtype=np.int64)

    # This variant crashes:
    # d = [{0:0.}] * n

    # This variant works:
    d = [{0:0.} for i in range(n)]



    for i in prange(len(x)):
        # just create some data
        y = (x[i]**1.2 + i**1.4)**0.9


        # Arguments of parDict.setitem(k, v)
        k = i
        v = y

        # split and fill fifos
        src_id = nb.get_thread_id()
        dst_id = hash(k)%n
        fifos_k[src_id,dst_id,fifo_idx[src_id,dst_id,0]] = k
        fifos_v[src_id,dst_id,fifo_idx[src_id,dst_id,0]] = v
        fifo_idx[src_id,dst_id,0] = (fifo_idx[src_id,dst_id,0] + 1) % fifo_cap

        # from fifos to dicts
        for src_id in range(n):
            dst_id = nb.get_thread_id()
            if fifo_idx[src_id,dst_id,1] == fifo_idx[src_id,dst_id,0]:
                continue
            k2 = fifos_k[src_id,dst_id,fifo_idx[src_id,dst_id,1]]
            v2 = fifos_v[src_id,dst_id,fifo_idx[src_id,dst_id,1]]
            fifo_idx[src_id,dst_id,1] = (fifo_idx[src_id,dst_id,1] + 1) % \
            fifo_cap
            d[dst_id][k2] = v2

    return d

nb.set_num_threads(16)

x = np.random.rand(100000000)
print(x[0])

print(par(x)[0])

