# Concept of numba dictionary parallelization

Currently `numba` lags the support of parallel dictionary writes. This approach shows the concept of an implementation without locking.

## Interface

Data can be written to the dict by:

```python
set_s(s, k, v)
```

The args: `k` and `v` are the known key and value of the dict. Because we have some threads running in parallel, `s` represents the thread number to write to one of `sn` different thread specific queues avoiding write race conditions. So `sn` represents the total number of parallel program threads.

Read operations are not critical. There is only a trivial read function:

```python
get(k, v)
```

The parameter `dn` defines the number of worker threads collecting the entries to wrote from the program thread queues. There are `dn` individual dicts to parallelize write operations. Which dict is selected, depends on the hash value of the key.

## Author

Stefan Helmert

