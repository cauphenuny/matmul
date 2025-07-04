import libmatmul as matmul
import os
import numpy as np
from timeit import timeit


def test_hello():
    pass


def test_lib():
    N = 1024
    a = np.random.rand(N, N).astype(np.int32)
    b = np.random.rand(N, N).astype(np.int32)
    c = matmul.trivial(a, b)
    c_ans = np.matmul(a, b)
    assert np.allclose(c, c_ans), "Matrix multiplication result is incorrect."
