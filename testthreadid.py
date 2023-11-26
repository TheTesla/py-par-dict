from numba import njit, prange
from numba.typed import List, Dict
import numba as nb
import numpy as np

@njit(parallel=True)
def par(x):
    print(nb.get_num_threads())
    print(nb.get_thread_id())
    d = [{0:0.}] * nb.get_num_threads()
    y = [0.]*len(x)
    for i in prange(len(x)):
        y[i] = (x[i]**1.2 + i**1.4)**0.9
        d[nb.get_thread_id()][i] = y[i]
    return y

x = np.random.rand(1000)
print(x[0])

print(par(x)[0])

