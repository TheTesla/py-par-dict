#!/usr/bin/env python3


from threading import Thread
from queue import Queue
from time import sleep

dn = 2
sn = 2

q_mat = [[Queue() for s in range(sn)] for d in range(dn)]

o_vec = [{} for d in range(dn)]

def get(k):
    return o_vec[hash(k)%len(o_vec)][k]

def set_d(k, v):
    o_vec[hash(k)%dn][k] = v

def write_d(q_mat, n):
    while(True):
        for s in range(sn):
            if not q_mat[n][s].empty():
                k, v = q_mat[n][s].get()
                set_d(k, v)
        sleep(0.1)

def set_s(s, k, v):
    q_mat[hash(k)%dn][s] = v


sleep(0.1)


t_vec = [Thread(target=write_d, args=(q_mat,d,)) for d in range(dn)]

for t in t_vec:
    t.start()


set_s(0, 'a', 3)
set_s(0, 'b', 4)

sleep(0.1)

print(get('a'))
print(get('b'))




