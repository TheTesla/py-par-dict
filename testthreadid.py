from numba import njit, prange
from numba.typed import List, Dict
import numba as nb
import numpy as np
import time

@njit(parallel=True)
def par(x):
    print(nb.get_num_threads())
    print(nb.get_thread_id())


    # initialize parDict
    n = nb.get_num_threads()
    fifo_cap = 2**20
    fifos_k = np.zeros((n,n,fifo_cap),dtype=np.int64)
    fifos_v = np.zeros((n,n,fifo_cap),dtype=np.float64)
    fifo_idx = np.zeros((n,n,2),dtype=np.int64)


    # initialize list of dicts with a dummy value to tell numba the datatype:
    ## This variant crashes:
    ## d = [{0:0.}] * n

    # This variant works:
    d = [{0:0.} for i in range(n)]

    # remove dummy value
    for e in d:
        del e[0]


    for i in prange(len(x)):
        # just create some data
        y = (x[i]**1.2 + i**1.4)**0.9

        # Arguments of parDict.setitem(k, v)
        k = i
        v = y

        # split and fill fifos
        src_id = nb.get_thread_id()
        dst_id = hash(k)%n
        #print((src_id,dst_id,k,fifo_idx[src_id,dst_id,0]))
        fifos_k[src_id,dst_id,fifo_idx[src_id,dst_id,0]] = k
        fifos_v[src_id,dst_id,fifo_idx[src_id,dst_id,0]] = v
        fifo_idx[src_id,dst_id,0] = (fifo_idx[src_id,dst_id,0] + 1) % fifo_cap

        # from fifos to dicts
        dst_id = nb.get_thread_id()
        for src_id in range(n):
            if fifo_idx[src_id,dst_id,1] == fifo_idx[src_id,dst_id,0]:
                continue
            k2 = fifos_k[src_id,dst_id,fifo_idx[src_id,dst_id,1]]
            v2 = fifos_v[src_id,dst_id,fifo_idx[src_id,dst_id,1]]
            fifo_idx[src_id,dst_id,1] = (fifo_idx[src_id,dst_id,1] + 1) % \
            fifo_cap

            #print((src_id,dst_id,k2,v2))
            d[dst_id][k2] = v2

    #return d

    # synchronize - write out all remaining data from the fifios to the dicts:
    print("sync:")
    for dst_id in prange(n):
        for src_id in range(n):
            while fifo_idx[src_id,dst_id,1] != fifo_idx[src_id,dst_id,0]:
                k2 = fifos_k[src_id,dst_id,fifo_idx[src_id,dst_id,1]]
                v2 = fifos_v[src_id,dst_id,fifo_idx[src_id,dst_id,1]]
                fifo_idx[src_id,dst_id,1] = (fifo_idx[src_id,dst_id,1] + 1) % \
                fifo_cap

                #print((src_id,dst_id,k2,v2))
                d[dst_id][k2] = v2

    return d

nb.set_num_threads(12)

x = np.random.rand(10000000)
print(len(x))
y = par(x)
l = [len(e) for e in y]
print(l)
print(sum(l))
#print(sorted([k for e in y for k,v in e.items()]))

