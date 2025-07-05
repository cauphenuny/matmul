import libmatmul as matmul
import os
import numpy as np
from timeit import timeit


def test_hello():
    pass

def run(func):
    N = 1024
    a = np.random.rand(N, N).astype(np.int32)
    b = np.random.rand(N, N).astype(np.int32)
    c = func(a, b)
    c_ans = np.matmul(a, b)
    assert np.allclose(c, c_ans), "Matrix multiplication result is incorrect."

def test_trivial():
    run(matmul.trivial)

def test_multithread():
    run(matmul.multithread)

def test_chunk():
    run(matmul.chunk)

def test_simd():
    run(matmul.auto_simd)